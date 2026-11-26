//! Synchronization primitives

#[cfg(target_arch = "x86_64")]
mod critical_section;
pub mod futex;
#[cfg(feature = "newlib")]
pub mod recmutex;
pub mod semaphore;
