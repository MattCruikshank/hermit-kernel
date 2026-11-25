//! Memory management.
//!
//! This is an overview of Hermit's memory layout:
//!
//! - `DeviceAlloc.device_offset` is 0 if `!cfg!(careful)`
//! - User space virtual memory is only used if `!cfg!(feature = "common-os")`
//! - On x86-64, PCI BARs, I/O APICs, and local APICs may be in `0xc0000000..0xffffffff`, which could be inside of `MEM`.
//!
//! ```text
//!                               Virtual address
//!                                    space
//!
//!                                 ...┌───┬──► 00000000
//!           Physical address   ...   │   │
//!                space      ...      │   │ Identity map
//!                        ...         │   │
//!    00000000 ◄──┬───┐...         ...├───┼──► mem_size
//!                │   │   ...   ...   │   │
//!     FrameAlloc │MEM│      ...      │   │ Unused
//!                │   │   ...   ...   │   │
//!    mem_size ◄──┼───┤...         ...├───┼──► DeviceAlloc.phys_offset
//!                │   │   ...         │   │
//!                │   │      ...      │   │ DeviceAlloc
//!                │   │         ...   │   │
//!          Empty │   │            ...├───┼──► DeviceAlloc.phys_offset + mem_size
//!                │   │               │   │
//!                │   │               │   │
//!                │   │               │   │ Unused
//!     Unknown ◄──┼───┤               │   │
//!                │   │               │   │
//!            PCI │   │               ├───┼──► kernel_virt_start
//!                │   │               │   │
//!     Unknown ◄──┼───┤               │   │ PageAlloc
//!                │   │               │   │
//!                │   │               ├───┼──► kernel_virt_end
//!                │   │               │   │
//!          Empty │   │               │   │
//!                │   │               │   │ User space
//!                │   │               │   │
//!                │   │               │   │
//! ```

pub(crate) mod device_alloc;
mod mapped_page_box;
mod page_range_alloc;
mod physicalmem;
mod virtualmem;

use core::alloc::Layout;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

use align_address::Align;
use free_list::{PageLayout, PageRange};
use hermit_sync::RawInterruptSpinMutex;
pub use memory_addresses::{PhysAddr, VirtAddr};
#[cfg(target_os = "none")]
use talc::TalcLock;
#[cfg(target_os = "none")]
use talc::base::Talc;
#[cfg(target_os = "none")]
use talc::base::binning::Binning;
#[cfg(target_os = "none")]
use talc::source::Source;

pub use self::mapped_page_box::MappedPageBox;
pub use self::page_range_alloc::{PageRangeAllocator, PageRangeBox};
pub use self::physicalmem::{FrameAlloc, FrameBox};
pub use self::virtualmem::{PageAlloc, PageBox};
use crate::arch;
#[cfg(any(target_arch = "x86_64", target_arch = "riscv64"))]
use crate::arch::mm::paging::HugePageSize;
pub use crate::arch::mm::paging::virtual_to_physical;
use crate::arch::mm::paging::{BasePageSize, LargePageSize, PageSize};

#[cfg(target_os = "none")]
#[global_allocator]
pub(crate) static ALLOCATOR: TalcLock<RawInterruptSpinMutex, GrowingHeap> =
	TalcLock::new(GrowingHeap::new());

/// Physical memory backing the heap at boot. The heap grows from there on demand.
#[cfg(not(feature = "common-os"))]
const INITIAL_HEAP_SIZE: usize = 16 << 20;
/// Minimum amount by which the heap grows when it runs out (rounded up to 2 MiB).
const HEAP_GROW_STEP: usize = 16 << 20;

/// Heap source that grows the heap on demand.
///
/// At boot, the heap's whole *virtual* range is reserved (as large as the available memory), but
/// only [`INITIAL_HEAP_SIZE`] of it is backed by frames. When talc runs out, [`Source::acquire`]
/// maps more frames at the end of the heap (2 MiB pages where possible, 4 KiB pages otherwise)
/// and extends the heap. The heap never shrinks. Memory that the heap doesn't use stays
/// available to `mmap` (e.g. Wasm linear memories), instead of a fixed share being set aside.
///
/// `acquire` runs with the allocator locked, so it must not allocate from the heap. It uses
/// [`FrameAlloc`] and the page tables, which don't — as long as the free lists stay within their
/// inline capacity (see `physicalmem.rs`).
#[derive(Debug)]
pub(crate) struct GrowingHeap {
	/// The heap end as returned by talc's `claim`/`extend`.
	heap_end: Option<NonNull<u8>>,
	/// End of the mapped (frame-backed) part of the heap.
	mapped_end: usize,
	/// End of the heap's virtual reservation.
	limit: usize,
}

// SAFETY: only accessed under the allocator's lock.
unsafe impl Send for GrowingHeap {}

impl GrowingHeap {
	pub const fn new() -> Self {
		Self {
			heap_end: None,
			mapped_end: 0,
			limit: 0,
		}
	}
}

/// Backs `size` bytes (a multiple of 2 MiB) at `virt` with frames, using 2 MiB pages where a
/// 2 MiB frame is free and 4 KiB pages otherwise. Returns how many bytes were mapped.
fn map_heap_range(virt: VirtAddr, size: usize) -> usize {
	let large = LargePageSize::SIZE as usize;
	let mut done = 0;
	while done < size {
		let at = virt + done as u64;
		if at.is_aligned_to(LargePageSize::SIZE)
			&& arch::kernel::processor::supports_2mib_pages()
			&& arch::mm::paging::map_heap::<LargePageSize>(at, 1).is_ok()
		{
			done += large;
			continue;
		}
		let pages = large / BasePageSize::SIZE as usize;
		match arch::mm::paging::map_heap::<BasePageSize>(at, pages) {
			Ok(()) => done += large,
			Err(mapped) => return done + mapped * BasePageSize::SIZE as usize,
		}
	}
	done
}

#[cfg(target_os = "none")]
unsafe impl Source for GrowingHeap {
	fn acquire<B: Binning>(talc: &mut Talc<Self, B>, layout: Layout) -> Result<(), ()> {
		let Some(heap_end) = talc.source.heap_end else {
			return Err(());
		};
		let large = LargePageSize::SIZE as usize;
		// Room for the request plus talc's bookkeeping, at least one grow step.
		let need = layout.size() + layout.align() + 256;
		let grow = need.max(HEAP_GROW_STEP).align_up(large);
		let start = talc.source.mapped_end;
		let grow = grow.min(talc.source.limit.saturating_sub(start));
		if grow == 0 {
			return Err(());
		}
		let mapped = map_heap_range(VirtAddr::new(start as u64), grow);
		if mapped == 0 {
			return Err(());
		}
		talc.source.mapped_end = start + mapped;
		// SAFETY: `heap_end` is talc's current heap end; the range up to `mapped_end` is now
		// mapped and owned by the heap.
		let new_end = unsafe { talc.extend(heap_end, talc.source.mapped_end as *mut u8) };
		talc.source.heap_end = Some(new_end);
		Ok(())
	}
}

#[cfg(target_os = "none")]
pub(crate) fn claim_initial_heap() {
	#[repr(C, align(0x1000))]
	struct InitialHeap([MaybeUninit<u8>; 0x1000]);

	debug_assert_eq!(
		Layout::new::<InitialHeap>(),
		Layout::from_size_align(0x1000, 0x1000).unwrap()
	);

	static mut INITIAL_HEAP: InitialHeap = InitialHeap([MaybeUninit::uninit(); _]);

	let base = (&raw mut INITIAL_HEAP).cast::<u8>();
	let size = size_of::<InitialHeap>();
	unsafe {
		ALLOCATOR.lock().claim(base, size).unwrap();
	}
}

#[cfg(target_os = "none")]
pub(crate) fn init() {
	use arch::mm::paging;

	#[cfg(not(target_arch = "riscv64"))]
	unsafe {
		paging::init();
	}
	unsafe {
		FrameAlloc::init();
	}
	#[cfg(target_arch = "x86_64")]
	unsafe {
		paging::log_page_tables();
	}
	unsafe {
		PageAlloc::init();
	}
	#[cfg(target_arch = "riscv64")]
	unsafe {
		paging::enable_page_table();
	}

	let total_mem = physicalmem::total_memory_size();
	info!("Total memory size: {} MiB", total_mem >> 20);

	// we reserve physical memory for the required page tables
	// In worst case, we use page size of BasePageSize::SIZE
	let npages = total_mem / BasePageSize::SIZE as usize;
	let npage_div = BasePageSize::SIZE as usize / align_of::<usize>();
	let npage_3tables = npages / npage_div + 1;
	let npage_2tables = npage_3tables / npage_div + 1;
	let npage_1tables = npage_2tables / npage_div + 1;
	let min_mem = (npage_3tables + npage_2tables + npage_1tables) * BasePageSize::SIZE as usize
		+ 2 * LargePageSize::SIZE as usize;
	#[cfg(any(target_arch = "x86_64", target_arch = "riscv64"))]
	let has_1gib_pages = arch::kernel::processor::supports_1gib_pages();
	let has_2mib_pages = arch::kernel::processor::supports_2mib_pages();

	info!("Minimum memory size: {} MiB", min_mem >> 20);
	let avail_mem = total_mem
		.checked_sub(min_mem)
		.unwrap_or_else(|| panic!("Not enough memory available!"))
		.align_down(LargePageSize::SIZE as usize);

	let mut map_addr;
	let mut map_size;
	let heap_start_addr;
	let heap_limit;

	#[cfg(feature = "common-os")]
	{
		info!("Using Hermit as common OS!");

		// we reserve at least 75% of the memory for the user space
		let reserve: usize = (avail_mem * 75) / 100;
		// 64 MB is enough as kernel heap
		let reserve = core::cmp::min(reserve, 0x0400_0000);

		let virt_size: usize = reserve.align_down(LargePageSize::SIZE as usize);
		let layout = PageLayout::from_size_align(virt_size, LargePageSize::SIZE as usize).unwrap();
		let page_range = PageAlloc::allocate(layout).unwrap();
		let virt_addr = VirtAddr::from(page_range.start());
		heap_start_addr = virt_addr;
		heap_limit = virt_addr + virt_size as u64;

		info!(
			"Heap: size {} MB, start address {:p}",
			virt_size >> 20,
			virt_addr
		);

		#[cfg(any(target_arch = "x86_64", target_arch = "riscv64"))]
		if has_1gib_pages && virt_size > HugePageSize::SIZE as usize {
			// Mount large pages to the next huge page boundary
			let npages = (virt_addr.align_up(HugePageSize::SIZE) - virt_addr) as usize
				/ LargePageSize::SIZE as usize;
			if let Err(n) = paging::map_heap::<LargePageSize>(virt_addr, npages) {
				map_addr = virt_addr + n as u64 * LargePageSize::SIZE;
				map_size = virt_size - (map_addr - virt_addr) as usize;
			} else {
				map_addr = virt_addr.align_up(HugePageSize::SIZE);
				map_size = virt_size - (map_addr - virt_addr) as usize;
			}
		} else {
			map_addr = virt_addr;
			map_size = virt_size;
		}

		#[cfg(not(any(target_arch = "x86_64", target_arch = "riscv64")))]
		{
			map_addr = virt_addr;
			map_size = virt_size;
		}
	}

	#[cfg(not(feature = "common-os"))]
	{
		// Reserve address space for a heap as large as the available memory, but back only
		// `INITIAL_HEAP_SIZE` of it now: the heap grows on demand (`GrowingHeap`), and memory
		// it doesn't use stays available for stacks and `mmap`. (Upstream sets aside a fixed
		// 75% with `mman`, 90% without.)
		let reserved: usize = avail_mem.align_down(LargePageSize::SIZE as usize);

		let layout = PageLayout::from_size_align(reserved, LargePageSize::SIZE as usize).unwrap();
		let page_range = PageAlloc::allocate(layout).unwrap();
		let virt_addr = VirtAddr::from(page_range.start());
		heap_start_addr = virt_addr;
		heap_limit = virt_addr + reserved as u64;
		let virt_size = reserved.min(INITIAL_HEAP_SIZE);

		info!(
			"Heap: {} MB backed initially, grows on demand up to {} MB, start address {:p}",
			virt_size >> 20,
			reserved >> 20,
			virt_addr
		);

		#[cfg(any(target_arch = "x86_64", target_arch = "riscv64"))]
		if has_1gib_pages && virt_size > HugePageSize::SIZE as usize {
			// Mount large pages to the next huge page boundary
			let npages = (virt_addr.align_up(HugePageSize::SIZE) - virt_addr) / LargePageSize::SIZE;
			if let Err(n) = paging::map_heap::<LargePageSize>(virt_addr, npages as usize) {
				map_addr = virt_addr + n as u64 * LargePageSize::SIZE;
				map_size = virt_size - (map_addr - virt_addr) as usize;
			} else {
				map_addr = virt_addr.align_up(HugePageSize::SIZE);
				map_size = virt_size - (map_addr - virt_addr) as usize;
			}
		} else {
			map_addr = virt_addr;
			map_size = virt_size;
		}

		#[cfg(not(any(target_arch = "x86_64", target_arch = "riscv64")))]
		{
			map_addr = virt_addr;
			map_size = virt_size;
		}
	}

	#[cfg(any(target_arch = "x86_64", target_arch = "riscv64"))]
	if has_1gib_pages
		&& map_size > HugePageSize::SIZE as usize
		&& map_addr.is_aligned_to(HugePageSize::SIZE)
	{
		let size = map_size.align_down(HugePageSize::SIZE as usize);
		if let Err(num_pages) =
			paging::map_heap::<HugePageSize>(map_addr, size / HugePageSize::SIZE as usize)
		{
			map_size -= num_pages * HugePageSize::SIZE as usize;
			map_addr += num_pages as u64 * HugePageSize::SIZE;
		} else {
			map_size -= size;
			map_addr += size;
		}
	}

	if has_2mib_pages
		&& map_size > LargePageSize::SIZE as usize
		&& map_addr.is_aligned_to(LargePageSize::SIZE)
	{
		let size = map_size.align_down(LargePageSize::SIZE as usize);
		if let Err(num_pages) =
			paging::map_heap::<LargePageSize>(map_addr, size / LargePageSize::SIZE as usize)
		{
			map_size -= num_pages * LargePageSize::SIZE as usize;
			map_addr += num_pages as u64 * LargePageSize::SIZE;
		} else {
			map_size -= size;
			map_addr += size;
		}
	}

	if map_size > BasePageSize::SIZE as usize && map_addr.is_aligned_to(BasePageSize::SIZE) {
		let size = map_size.align_down(BasePageSize::SIZE as usize);
		if let Err(num_pages) =
			paging::map_heap::<BasePageSize>(map_addr, size / BasePageSize::SIZE as usize)
		{
			map_size -= num_pages * BasePageSize::SIZE as usize;
			map_addr += num_pages as u64 * BasePageSize::SIZE;
		} else {
			map_size -= size;
			map_addr += size;
		}
	}

	let heap_end_addr = map_addr;

	let size = heap_end_addr.as_usize() - heap_start_addr.as_usize();
	{
		let mut talc = ALLOCATOR.lock();
		let heap_end = unsafe { talc.claim(heap_start_addr.as_mut_ptr(), size).unwrap() };
		talc.source = GrowingHeap {
			heap_end: Some(heap_end),
			mapped_end: heap_end_addr.as_usize(),
			limit: heap_limit.as_usize(),
		};
	}

	info!("Heap is located at {heap_start_addr:p}..{heap_end_addr:p} ({map_size} Bytes unmapped)");
}

pub(crate) fn print_information() {
	debug!("{FrameAlloc}");
	debug!("{PageAlloc}");
}

/// Maps a given physical address and size in virtual space and returns address.
#[cfg(feature = "pci")]
pub(crate) fn map(
	physical_address: PhysAddr,
	size: usize,
	writable: bool,
	no_execution: bool,
	no_cache: bool,
) -> VirtAddr {
	use crate::arch::mm::paging::PageTableEntryFlags;
	#[cfg(target_arch = "x86_64")]
	use crate::arch::mm::paging::PageTableEntryFlagsExt;

	let size = size.align_up(BasePageSize::SIZE as usize);
	let count = size / BasePageSize::SIZE as usize;

	let mut flags = if no_cache {
		PageTableEntryFlags::device()
	} else {
		PageTableEntryFlags::normal()
	};
	flags.set_writable(writable).set_execute(!no_execution);

	let layout = PageLayout::from_size(size).unwrap();
	let page_range = PageAlloc::allocate(layout).unwrap();
	let virtual_address = VirtAddr::from(page_range.start());
	arch::mm::paging::map::<BasePageSize>(virtual_address, physical_address, count, flags);

	virtual_address
}

#[allow(dead_code)]
/// Unmaps virtual address, without 'freeing' physical memory it is mapped to!
pub(crate) fn unmap(virtual_address: VirtAddr, size: usize) {
	let size = size.align_up(BasePageSize::SIZE as usize);

	if virtual_to_physical(virtual_address).is_some() {
		arch::mm::paging::unmap::<BasePageSize>(
			virtual_address,
			size / BasePageSize::SIZE as usize,
		);

		let range = PageRange::from_start_len(virtual_address.as_usize(), size).unwrap();
		unsafe {
			PageAlloc::deallocate(range);
		}
	} else {
		panic!("No page table entry for virtual address {virtual_address:p}");
	}
}
