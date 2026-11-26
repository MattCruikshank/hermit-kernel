//! A [`critical_section`] implementation: interrupts off on this core, then one global lock,
//! re-entrant on the same core.
//!
//! Why: `event-listener` (under `async-lock`, which the kernel uses for its file descriptors)
//! otherwise protects its listener lists with a plain spinlock that leaves interrupts on. With
//! preemptive scheduling, a task could be preempted while holding one, and other cores then spun
//! on it forever (the task can't run again while its own core spins: a whole-VM hang, METAL
//! ISSUE-00053). In a critical section the holder can't be preempted, and the sections are a few
//! pointer updates long, so waiting for one is bounded.

use core::hint::spin_loop;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::arch::kernel::core_local::core_id;

/// The core inside the critical section (`u32::MAX`: none), and how deep (only that core
/// touches `DEPTH`).
static OWNER: AtomicU32 = AtomicU32::new(u32::MAX);
static DEPTH: AtomicU32 = AtomicU32::new(0);

struct HermitCriticalSection;
critical_section::set_impl!(HermitCriticalSection);

// SAFETY: acquire disables interrupts before taking the lock and release restores them after
// giving it up, so a holder is never preempted; nesting on one core only counts.
unsafe impl critical_section::Impl for HermitCriticalSection {
    unsafe fn acquire() -> critical_section::RawRestoreState {
        let was_enabled = x86_64::instructions::interrupts::are_enabled();
        x86_64::instructions::interrupts::disable();
        let me = core_id();
        if OWNER.load(Ordering::Acquire) == me {
            DEPTH.fetch_add(1, Ordering::Relaxed);
        } else {
            while OWNER
                .compare_exchange_weak(u32::MAX, me, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                spin_loop();
            }
            DEPTH.store(1, Ordering::Relaxed);
        }
        was_enabled
    }

    unsafe fn release(was_enabled: critical_section::RawRestoreState) {
        if DEPTH.fetch_sub(1, Ordering::Relaxed) == 1 {
            OWNER.store(u32::MAX, Ordering::Release);
        }
        if was_enabled {
            x86_64::instructions::interrupts::enable();
        }
    }
}
