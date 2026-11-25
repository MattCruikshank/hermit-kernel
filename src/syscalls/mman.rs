//! Memory management syscalls.
//!
//! These system calls are similar to [sys/mman.h].
//!
//! <div class="warning">These system calls are not very POSIX-like yet!</div>
//!
//! [sys/mman.h]: https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/sys_mman.h.html

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

/// Creates a new virtual memory mapping of the `size` specified with
/// protection bits specified in `prot_flags`.
#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_mmap(size: usize, prot_flags: MemoryProtection, ret: &mut *mut u8) -> i32 {
	let size = size.align_up(BasePageSize::SIZE as usize);
	let layout = PageLayout::from_size(size).unwrap();
	let page_range = PageAlloc::allocate(layout).unwrap();
	let virtual_address = VirtAddr::from(page_range.start());
	if prot_flags.is_empty() {
		*ret = virtual_address.as_mut_ptr();
		return 0;
	}
	let frame_layout = PageLayout::from_size(size).unwrap();
	let frame_range = FrameAlloc::allocate_zeroed(frame_layout).unwrap();
	let physical_address = PhysAddr::from(frame_range.start());

	debug!("Mmap {physical_address:X} -> {virtual_address:X} ({size})");
	let count = size / BasePageSize::SIZE as usize;
	let mut flags = PageTableEntryFlags::normal();
	flags
		.set_writable(prot_flags.contains(MemoryProtection::Write))
		.set_execute(prot_flags.contains(MemoryProtection::Exec));

	paging::map::<BasePageSize>(virtual_address, physical_address, count, flags);

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

	// Unmap and free the frames run by run, as the range may consist of several mappings.
	let mut virt = virtual_address;
	let mut remaining = size / PAGE_SIZE;
	while remaining > 0 {
		let (count, phys) = contiguous_pages(virt, remaining);
		if let Some(phys) = phys {
			paging::unmap::<BasePageSize>(virt, count);
			debug!("Unmapping {virt:X} ({count} pages) -> {phys:X}");

			let frame_range =
				PageRange::from_start_len(phys.as_u64() as usize, count * PAGE_SIZE).unwrap();
			unsafe {
				FrameAlloc::deallocate(frame_range);
			}
		}
		virt += (count * PAGE_SIZE) as u64;
		remaining -= count;
	}

	unsafe {
		PageAlloc::deallocate(page_range);
	}

	0
}

/// Configures the protections associated with a region of virtual memory
/// starting at `ptr` and going to `size`.
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
	let mut flags = PageTableEntryFlags::normal();
	flags
		.set_writable(prot_flags.contains(MemoryProtection::Write))
		.set_execute(prot_flags.contains(MemoryProtection::Exec));

	let virtual_address = VirtAddr::from_ptr(ptr);
	debug!("Mprotect {virtual_address:X} ({size}) -> {prot_flags:?})");

	// The range may consist of several runs of mapped and unmapped pages (e.g. a reservation
	// that was made accessible piece by piece), so handle it run by run. Check first, so that a
	// failing call changes nothing.
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
	while remaining > 0 {
		let (run, mapped) = contiguous_pages(virt, remaining);
		if let Some(physical_address) = mapped {
			// Memory is already mapped: change its flags.
			paging::map::<BasePageSize>(virt, physical_address, run, flags);
		} else if prot_flags != MemoryProtection::None {
			// Reserved but unmapped memory becomes accessible: back it with zeroed frames.
			let frame_layout = PageLayout::from_size(run * PAGE_SIZE).unwrap();
			let frame_range = FrameAlloc::allocate_zeroed(frame_layout).unwrap();
			let physical_address = PhysAddr::from(frame_range.start());
			paging::map::<BasePageSize>(virt, physical_address, run, flags);
		}
		// Unmapped memory that stays inaccessible stays reserved without frames.
		virt += (run * PAGE_SIZE) as u64;
		remaining -= run;
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
