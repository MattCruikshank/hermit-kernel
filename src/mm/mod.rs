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

use core::alloc::{GlobalAlloc, Layout};
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use align_address::Align;
use free_list::{PageLayout, PageRange};
use hermit_sync::{InterruptTicketMutex, InterruptTicketMutexGuard, RawInterruptSpinMutex};
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

/// A mutex that records which core holds it, so that the heap can tell whether growing it here
/// would wait for a lock this core already holds (ISSUE-00015). Interrupts are disabled while
/// it is held, so "held by this core" means "held by the code we were called from".
pub(crate) struct OwnedMutex<T> {
	inner: InterruptTicketMutex<T>,
	owner: AtomicU32,
}

const NO_OWNER: u32 = u32::MAX;

impl<T> OwnedMutex<T> {
	pub const fn new(value: T) -> Self {
		Self {
			inner: InterruptTicketMutex::new(value),
			owner: AtomicU32::new(NO_OWNER),
		}
	}

	pub fn lock(&self) -> OwnedMutexGuard<'_, T> {
		let guard = self.inner.lock();
		self.owner
			.store(arch::kernel::core_local::core_id(), Ordering::Relaxed);
		OwnedMutexGuard {
			guard,
			owner: &self.owner,
		}
	}

	/// Whether this core holds the lock. Only valid once per-core data is set up.
	pub fn held_by_this_core(&self) -> bool {
		self.owner.load(Ordering::Relaxed) == arch::kernel::core_local::core_id()
	}
}

pub(crate) struct OwnedMutexGuard<'a, T> {
	guard: InterruptTicketMutexGuard<'a, T>,
	owner: &'a AtomicU32,
}

impl<T> Drop for OwnedMutexGuard<'_, T> {
	fn drop(&mut self) {
		// Runs before `guard` is dropped, i.e. while the lock is still held.
		self.owner.store(NO_OWNER, Ordering::Relaxed);
	}
}

impl<T> Deref for OwnedMutexGuard<'_, T> {
	type Target = T;
	fn deref(&self) -> &T {
		&self.guard
	}
}

impl<T> DerefMut for OwnedMutexGuard<'_, T> {
	fn deref_mut(&mut self) -> &mut T {
		&mut self.guard
	}
}

/// The heap allocator. `GlobalAlloc` goes through [`KernelAllocator`], which maps more heap memory
/// when needed — outside the allocator lock (ISSUE-00015).
#[cfg(target_os = "none")]
pub(crate) static TALC: TalcLock<RawInterruptSpinMutex, GrowingHeap> =
	TalcLock::new(GrowingHeap::new());

#[cfg(target_os = "none")]
#[global_allocator]
pub(crate) static ALLOCATOR: KernelAllocator = KernelAllocator;

/// Physical memory given to the heap at boot. The heap grows from there on demand.
#[cfg(not(feature = "common-os"))]
const INITIAL_HEAP_SIZE: usize = 16 << 20;
/// Pre-mapped memory kept beyond the heap's end, so that the heap can grow without mapping
/// anything while the allocator is locked.
const HEAP_HEADROOM: usize = 16 << 20;
/// Minimum amount by which the heap grows into the headroom when it runs out.
const HEAP_GROW_STEP: usize = 4 << 20;

/// Set by `acquire` when the headroom ran low; the next allocation outside the lock refills it.
#[cfg(target_os = "none")]
static REFILL_WANTED: AtomicBool = AtomicBool::new(false);
/// One refill at a time.
#[cfg(target_os = "none")]
static REFILLING: AtomicBool = AtomicBool::new(false);
/// Set once `init` has set up the heap (before that, no refills; `core_id` may not work yet).
#[cfg(target_os = "none")]
static HEAP_READY: AtomicBool = AtomicBool::new(false);

/// Heap source that grows the heap on demand — without taking any other lock.
///
/// The heap's whole *virtual* range is reserved at boot (as large as the available memory).
/// Frames are mapped only for the part in use plus [`HEAP_HEADROOM`]. When talc runs out,
/// [`Source::acquire`] — which runs with the allocator locked — merely extends the heap into the
/// already-mapped headroom. Mapping more headroom ([`refill`]) needs the frame allocator and the
/// page tables, whose free lists may themselves allocate from the heap (`SmallVec` spill), so it
/// runs *outside* the allocator lock and never on a core that holds a free-list lock. Taking
/// those locks under the allocator lock could deadlock (ISSUE-00015).
///
/// The heap never shrinks (ISSUE-00016). Memory the heap doesn't use stays available to `mmap`.
#[derive(Debug)]
pub(crate) struct GrowingHeap {
	/// Start of the heap.
	start: usize,
	/// The heap end as returned by talc's `claim`/`extend`.
	heap_end: Option<NonNull<u8>>,
	/// End of the part given to talc (page aligned).
	talc_end: usize,
	/// End of the mapped (frame-backed) part: `talc_end..mapped_end` is the headroom.
	mapped_end: usize,
	/// End of the heap's virtual reservation.
	limit: usize,
}

// SAFETY: only accessed under the allocator's lock.
unsafe impl Send for GrowingHeap {}

impl GrowingHeap {
	pub const fn new() -> Self {
		Self {
			start: 0,
			heap_end: None,
			talc_end: 0,
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
		// Runs with the allocator locked: no frame allocation, no page-table changes, no locks.
		let Some(heap_end) = talc.source.heap_end else {
			return Err(());
		};
		let headroom = talc.source.mapped_end - talc.source.talc_end;
		// Room for the request plus talc's bookkeeping.
		let need = (layout.size() + layout.align() + 256).align_up(BasePageSize::SIZE as usize);
		if need > headroom {
			// The caller (`KernelAllocator`) maps more and retries.
			REFILL_WANTED.store(true, Ordering::Relaxed);
			return Err(());
		}
		let give = need.max(HEAP_GROW_STEP).min(headroom);
		talc.source.talc_end += give;
		// SAFETY: `heap_end` is talc's current heap end; `..talc_end` is mapped and unused.
		let new_end = unsafe { talc.extend(heap_end, talc.source.talc_end as *mut u8) };
		talc.source.heap_end = Some(new_end);
		if headroom - give < HEAP_HEADROOM / 2 {
			REFILL_WANTED.store(true, Ordering::Relaxed);
		}
		Ok(())
	}
}

#[cfg(target_os = "none")]
enum Refill {
	/// More memory was mapped.
	Done,
	/// Another core is refilling right now.
	Busy,
	/// Not possible here (inside a free-list lock, heap not ready, limit or memory exhausted).
	No,
}

/// Maps more headroom: at least `min_extra` bytes, and up to [`HEAP_HEADROOM`] in total.
/// Only called by [`KernelAllocator`], never with the allocator lock held.
#[cfg(target_os = "none")]
fn refill(min_extra: usize) -> Refill {
	if !HEAP_READY.load(Ordering::Acquire)
		|| physicalmem::held_by_this_core()
		|| virtualmem::held_by_this_core()
	{
		return Refill::No;
	}
	if REFILLING
		.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
		.is_err()
	{
		return Refill::Busy;
	}
	// Interrupts off: an interrupt handler on this core must not find `REFILLING` set by the code
	// it interrupted (it would spin on `Busy` forever).
	let mapped = hermit_sync::without_interrupts(|| {
		let (start, headroom, limit) = {
			let talc = TALC.lock();
			let s = &talc.source;
			(s.mapped_end, s.mapped_end - s.talc_end, s.limit)
		};
		let want = HEAP_HEADROOM
			.saturating_sub(headroom)
			.max(min_extra)
			.align_up(LargePageSize::SIZE as usize)
			.min(limit - start);
		if want == 0 {
			return 0;
		}
		// The free lists may allocate from the heap in here: talc's lock is free, and an
		// allocation that would need to grow the heap fails instead of recursing (this core then
		// holds a free-list lock, so `refill` returns `No`).
		let mapped = map_heap_range(VirtAddr::new(start as u64), want);
		TALC.lock().source.mapped_end = start + mapped;
		mapped
	});
	REFILL_WANTED.store(false, Ordering::Relaxed);
	REFILLING.store(false, Ordering::Release);
	if mapped > 0 { Refill::Done } else { Refill::No }
}

/// Memory statistics (METAL): `[total RAM, free RAM, heap mapped, heap given to the allocator,
/// heap in use, live heap allocations]`, in bytes except the last.
#[cfg(target_os = "none")]
pub(crate) fn metal_mem_info() -> [usize; 6] {
	// Free RAM first: the free-list lock may be held while the allocator lock is taken (a list
	// spilling to the heap), never the other way round.
	let free = FrameAlloc::free_bytes();
	let talc = TALC.lock();
	let s = &talc.source;
	let counters = talc.counters();
	[
		physicalmem::total_memory_size(),
		free,
		s.mapped_end.saturating_sub(s.start),
		s.talc_end.saturating_sub(s.start),
		counters.allocated_bytes,
		counters.allocation_count,
	]
}

/// The global allocator: talc, plus heap growth outside its lock.
#[cfg(target_os = "none")]
pub(crate) struct KernelAllocator;

#[cfg(target_os = "none")]
impl KernelAllocator {
	/// After an allocation attempt: refill the headroom if it ran low, or — if the attempt
	/// failed — map enough for `size` bytes and retry. Returns null instead of waiting wherever
	/// waiting could deadlock; a failed allocation then aborts with an error, it never hangs.
	#[inline]
	fn finish(&self, ptr: *mut u8, size: usize, retry: impl Fn() -> *mut u8) -> *mut u8 {
		if !ptr.is_null() {
			if REFILL_WANTED.load(Ordering::Relaxed) {
				let _ = refill(0);
			}
			return ptr;
		}
		loop {
			match refill(size.saturating_add(4096)) {
				Refill::Done => {
					let ptr = retry();
					if !ptr.is_null() {
						return ptr;
					}
				}
				// Another core is mapping memory. It doesn't wait for anything this core holds:
				// if this core held a free-list lock, `refill` would have returned `No`.
				Refill::Busy => {
					core::hint::spin_loop();
					let ptr = retry();
					if !ptr.is_null() {
						return ptr;
					}
				}
				Refill::No => return ptr::null_mut(),
			}
		}
	}
}

#[cfg(target_os = "none")]
unsafe impl GlobalAlloc for KernelAllocator {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		let ptr = unsafe { TALC.alloc(layout) };
		self.finish(ptr, layout.size(), || unsafe { TALC.alloc(layout) })
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		let ptr = unsafe { TALC.alloc_zeroed(layout) };
		self.finish(ptr, layout.size(), || unsafe { TALC.alloc_zeroed(layout) })
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { TALC.dealloc(ptr, layout) }
	}

	unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
		let new = unsafe { TALC.realloc(ptr, layout, new_size) };
		self.finish(new, new_size, || unsafe { TALC.realloc(ptr, layout, new_size) })
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
		TALC.lock().claim(base, size).unwrap();
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
		// Map the initial heap plus the headroom; talc gets only the initial part.
		let virt_size = reserved.min(INITIAL_HEAP_SIZE + HEAP_HEADROOM);

		info!(
			"Heap: {} MB mapped initially (incl. headroom), grows on demand up to {} MB, start address {:p}",
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

	let mapped_size = heap_end_addr.as_usize() - heap_start_addr.as_usize();
	#[cfg(not(feature = "common-os"))]
	let size = mapped_size.min(INITIAL_HEAP_SIZE);
	#[cfg(feature = "common-os")]
	let size = mapped_size;
	{
		let mut talc = TALC.lock();
		let heap_end = unsafe { talc.claim(heap_start_addr.as_mut_ptr(), size).unwrap() };
		talc.source = GrowingHeap {
			start: heap_start_addr.as_usize(),
			heap_end: Some(heap_end),
			talc_end: heap_start_addr.as_usize() + size,
			mapped_end: heap_end_addr.as_usize(),
			limit: heap_limit.as_usize(),
		};
	}
	HEAP_READY.store(true, Ordering::Release);

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
