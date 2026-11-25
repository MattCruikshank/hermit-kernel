//! Memory management syscalls.
//!
//! These system calls are similar to [sys/mman.h].
//!
//! <div class="warning">These system calls are not very POSIX-like yet!</div>
//!
//! [sys/mman.h]: https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/sys_mman.h.html
use alloc::vec::Vec;

use core::ffi::{c_int, c_void};

use align_address::Align;
use free_list::{PageLayout, PageRange};
use memory_addresses::{PhysAddr, VirtAddr};

#[cfg(target_arch = "x86_64")]
use crate::arch::mm::paging::PageTableEntryFlagsExt;
use crate::arch::mm::paging::{self, BasePageSize, PageSize, PageTableEntryFlags};
use crate::errno::Errno;
use crate::mm::{FrameAlloc, PageAlloc, PageRangeAllocator};

bitflags! {
	#[repr(transparent)]
	#[derive(Debug, Copy, Clone, Default, PartialEq)]
	pub struct MemoryProtection: u32 {
		/// Pages may not be accessed.
		const None = 0;
		/// Indicates that the memory region should be readable.
		const Read = 1 << 0;
		/// Indicates that the memory region should be writable.
		const Write = 1 << 1;
		/// Indicates that the memory region should be executable.
		const Exec = 1 << 2;
	}
}

const PAGE_SIZE: usize = BasePageSize::SIZE as usize;

/// Physical memory that `mmap`/`mprotect` leave free for kernel heap growth.
const HEAP_RESERVE: usize = 32 << 20;

/// Looks at up to `count` pages starting at `virtual_address` and returns how many of them,
/// from the start, share the state of the first page, together with the first page's frame.
///
/// Sharing the state means: all pages are unmapped (frame is `None`), or all pages are mapped
/// to physically contiguous frames.
fn contiguous_pages(virtual_address: VirtAddr, count: usize) -> (usize, Option<PhysAddr>) {
	let first_frame = paging::virtual_to_physical(virtual_address);

	let mut len = 1;
	while len < count {
		let offset = (len * PAGE_SIZE) as u64;
		let frame = paging::virtual_to_physical(virtual_address + offset);
		let expected_frame = first_frame.map(|phys| phys + offset);
		if frame != expected_frame {
			break;
		}
		len += 1;
	}

	(len, first_frame)
}

/// Backs `count` reserved pages starting at `virtual_address` with zeroed frames.
///
/// Physical memory doesn't have to be contiguous: the range is backed in chunks, starting with
/// one chunk for everything and halving the chunk size whenever no contiguous run that large is
/// free, down to single pages. So a fragmented physical memory doesn't make large mappings fail
/// while enough memory is free. On failure, everything mapped here is released again.
fn map_zeroed(
	virtual_address: VirtAddr,
	count: usize,
	flags: PageTableEntryFlags,
) -> Result<(), Errno> {
	// Keep a reserve for the kernel heap, which grows on demand (`mm::GrowingHeap`): if mappings
	// (e.g. Wasm linear memories) could take all frames, the next heap allocation that needs to
	// grow the heap would fail, and a failed heap allocation aborts.
	if FrameAlloc::free_bytes() < count * PAGE_SIZE + HEAP_RESERVE {
		return Err(Errno::Nomem);
	}
	let mut done = 0;
	let mut chunk = count;
	while done < count {
		chunk = chunk.min(count - done);
		let layout = PageLayout::from_size(chunk * PAGE_SIZE).unwrap();
		match FrameAlloc::allocate_zeroed(layout) {
			Ok(frame_range) => {
				let virt = virtual_address + (done * PAGE_SIZE) as u64;
				let phys = PhysAddr::from(frame_range.start());
				paging::map::<BasePageSize>(virt, phys, chunk, flags);
				done += chunk;
			}
			Err(_) if chunk > 1 => chunk /= 2,
			Err(_) => {
				release_frames(virtual_address, done);
				return Err(Errno::Nomem);
			}
		}
	}
	Ok(())
}

/// Unmaps the mapped pages among `count` pages starting at `virtual_address` and frees their
/// frames. The virtual range itself stays allocated (reserved).
fn release_frames(virtual_address: VirtAddr, count: usize) {
	// Unmap run by run (the range may consist of several mappings), then make sure no core
	// still translates to the old frames, and only then free them (METAL ISSUE-00032: other
	// cores' stale TLB entries let them use frames that had been given to someone else).
	let mut freed = Vec::new();
	let mut virt = virtual_address;
	let mut remaining = count;
	while remaining > 0 {
		let (run, phys) = contiguous_pages(virt, remaining);
		if let Some(phys) = phys {
			paging::unmap::<BasePageSize>(virt, run);
			debug!("Unmapping {virt:X} ({run} pages) -> {phys:X}");
			freed.push(PageRange::from_start_len(phys.as_u64() as usize, run * PAGE_SIZE).unwrap());
		}
		virt += (run * PAGE_SIZE) as u64;
		remaining -= run;
	}
	if freed.is_empty() {
		return;
	}
	crate::arch::kernel::apic::tlb_shootdown();
	for frame_range in freed {
		unsafe {
			FrameAlloc::deallocate(frame_range);
		}
	}
}

fn page_table_flags(prot_flags: MemoryProtection) -> PageTableEntryFlags {
	let mut flags = PageTableEntryFlags::normal();
	flags
		.set_writable(prot_flags.contains(MemoryProtection::Write))
		.set_execute(prot_flags.contains(MemoryProtection::Exec));
	flags
}

/// Creates a new virtual memory mapping of the `size` specified with
/// protection bits specified in `prot_flags`.
///
/// With [`MemoryProtection::None`], only address space is reserved (no frames). Otherwise the
/// mapping is backed right away (there is no demand paging), by frames that need not be
/// physically contiguous. Fails with `ENOMEM` instead of panicking when memory runs out.
#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_mmap(size: usize, prot_flags: MemoryProtection, ret: &mut *mut u8) -> i32 {
	let size = size.align_up(BasePageSize::SIZE as usize);
	let layout = PageLayout::from_size(size).unwrap();
	let Ok(page_range) = PageAlloc::allocate(layout) else {
		return -i32::from(Errno::Nomem);
	};
	let virtual_address = VirtAddr::from(page_range.start());
	if !prot_flags.is_empty() {
		let count = size / PAGE_SIZE;
		if let Err(errno) = map_zeroed(virtual_address, count, page_table_flags(prot_flags)) {
			unsafe {
				PageAlloc::deallocate(page_range);
			}
			return -i32::from(errno);
		}
		debug!("Mmap {virtual_address:X} ({size})");
	}
	*ret = virtual_address.as_mut_ptr();
	0
}

/// Unmaps memory at the specified `ptr` for `size` bytes.
#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_munmap(ptr: *mut u8, size: usize) -> i32 {
	let virtual_address = VirtAddr::from_ptr(ptr);
	let size = size.align_up(BasePageSize::SIZE as usize);
	let page_range = PageRange::from_start_len(virtual_address.as_usize(), size).unwrap();

	release_frames(virtual_address, size / PAGE_SIZE);

	unsafe {
		PageAlloc::deallocate(page_range);
	}

	0
}

/// Releases the physical memory behind `size` bytes at `ptr` (from [`sys_mmap`]) while keeping
/// the address range reserved, like `madvise(MADV_DONTNEED)` followed by `PROT_NONE`: the pages
/// become inaccessible and read as zeros once made accessible again with [`sys_mprotect`].
///
/// Hermit has no `MAP_FIXED`; this is how a mapping can give memory back without giving up its
/// addresses (e.g. Wasmtime's pooling allocator resetting a linear-memory slot).
#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_mdecommit(ptr: *mut u8, size: usize) -> i32 {
	let size = size.align_up(PAGE_SIZE);
	release_frames(VirtAddr::from_ptr(ptr), size / PAGE_SIZE);
	0
}

/// Configures the protections associated with a region of virtual memory
/// starting at `ptr` and going to `size`.
///
/// Reserved pages that become accessible are backed with zeroed frames. Making mapped memory
/// inaccessible ([`MemoryProtection::None`]) is not supported (`ENOSYS`); use [`sys_mdecommit`]
/// if the contents may be discarded.
///
/// Returns 0 on success and an error code on failure.
#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_mprotect(ptr: *mut u8, size: usize, prot_flags: MemoryProtection) -> i32 {
	let size = size.align_up(PAGE_SIZE);
	let count = size / PAGE_SIZE;
	if count == 0 {
		return 0;
	}
	let flags = page_table_flags(prot_flags);
	let virtual_address = VirtAddr::from_ptr(ptr);
	debug!("Mprotect {virtual_address:X} ({size}) -> {prot_flags:?})");

	// The range may consist of several runs of mapped and unmapped pages (e.g. a reservation
	// that was made accessible piece by piece), so handle it run by run. Check first, so that an
	// unsupported request changes nothing.
	let mut virt = virtual_address;
	let mut remaining = count;
	while remaining > 0 {
		let (run, mapped) = contiguous_pages(virt, remaining);
		if mapped.is_some() && prot_flags == MemoryProtection::None {
			// Making mapped memory inaccessible would need non-present entries that still own
			// their frames; not supported yet.
			return -i32::from(Errno::Nosys);
		}
		virt += (run * PAGE_SIZE) as u64;
		remaining -= run;
	}

	let mut virt = virtual_address;
	let mut remaining = count;
	let mut changed_mapped = false;
	while remaining > 0 {
		let (run, mapped) = contiguous_pages(virt, remaining);
		if let Some(physical_address) = mapped {
			// Memory is already mapped: change its flags.
			paging::map::<BasePageSize>(virt, physical_address, run, flags);
			changed_mapped = true;
		} else if prot_flags != MemoryProtection::None {
			// Reserved memory becomes accessible: back it with zeroed frames. On ENOMEM, runs
			// before this one keep their new protection.
			if let Err(errno) = map_zeroed(virt, run, flags) {
				return -i32::from(errno);
			}
		}
		// Reserved memory that stays inaccessible stays reserved without frames.
		virt += (run * PAGE_SIZE) as u64;
		remaining -= run;
	}
	// METAL (ISSUE-00033): `paging::map` only asks other cores to flush, without waiting. Wait
	// here, so no core keeps e.g. a writable translation to memory that's now read-only.
	if changed_mapped {
		crate::arch::kernel::apic::tlb_shootdown();
	}
	0
}

#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_mlock(_addr: *const c_void, _size: usize) -> i32 {
	// Hermit does not do any swapping yet.
	0
}

#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_munlock(_addr: *const c_void, _size: usize) -> i32 {
	// Hermit does not do any swapping yet.
	0
}

#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_mlockall(_flags: c_int) -> i32 {
	// Hermit does not do any swapping yet.
	0
}

#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_munlockall(_flags: c_int) -> i32 {
	// Hermit does not do any swapping yet.
	0
}
