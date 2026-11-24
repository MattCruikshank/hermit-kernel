#[cfg(feature = "tcp")]
pub(crate) mod tcp;
#[cfg(feature = "udp")]
pub(crate) mod udp;
#[cfg(feature = "virtio-vsock")]
pub(crate) mod vsock;

/// Returns a port from the IANA dynamic/ephemeral range (49152..=65535), cycling through it.
///
/// Used for TCP connects and for binding sockets to port 0 ("any port").
#[cfg(any(feature = "tcp", feature = "udp"))]
pub(crate) fn get_ephemeral_port() -> u16 {
	use core::sync::atomic::{AtomicU16, Ordering};

	const FIRST: u16 = 49152;
	static NEXT: AtomicU16 = AtomicU16::new(0);

	FIRST + NEXT.fetch_add(1, Ordering::Relaxed) % (u16::MAX - FIRST + 1)
}
