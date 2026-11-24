use alloc::boxed::Box;
use alloc::collections::vec_deque::VecDeque;
use core::future::{self, Future};
use core::mem;
use core::pin::pin;
use core::task::{Poll, Waker, ready};

use async_lock::Mutex;
use async_trait::async_trait;

use crate::errno::Errno;
use crate::fd::{EventFlags, ObjectInterface, PollEvent};
use crate::io;

#[derive(Debug)]
struct EventState {
	pub counter: u64,
	pub read_queue: VecDeque<Waker>,
	pub write_queue: VecDeque<Waker>,
}

impl EventState {
	pub fn new(counter: u64) -> Self {
		Self {
			counter,
			read_queue: VecDeque::new(),
			write_queue: VecDeque::new(),
		}
	}
}

pub(crate) struct EventFd {
	state: Mutex<EventState>,
	flags: EventFlags,
}

impl EventFd {
	pub fn new(initval: u64, flags: EventFlags) -> Self {
		debug!("Create EventFd {initval}, {flags:?}");
		Self {
			state: Mutex::new(EventState::new(initval)),
			flags,
		}
	}
}

/// Enqueues the waker unless an equivalent one is already queued. `poll`-based multiplexing
/// registers on every poll, so without this the queue grows with stale wakers.
fn enqueue(queue: &mut VecDeque<Waker>, waker: &Waker) {
	if !queue.iter().any(|w| w.will_wake(waker)) {
		queue.push_back(waker.clone());
	}
}

/// Wakes every queued waker. Waking only one (the oldest) lost wakeups: the oldest entry is
/// often a stale waker left behind by a `poll` that already returned for another reason (e.g.
/// a timeout), so the task actually waiting was never woken. (Upstream fixed this in 4820ad9e.)
fn wake_all(queue: &mut VecDeque<Waker>) {
	for waker in mem::take(queue) {
		waker.wake();
	}
}

#[async_trait]
impl ObjectInterface for EventFd {
	async fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
		let len = mem::size_of::<u64>();

		if buf.len() < len {
			return Err(Errno::Inval);
		}

		future::poll_fn(|cx| {
			if self.flags.contains(EventFlags::EFD_SEMAPHORE) {
				let mut pinned = pin!(self.state.lock());
				let mut guard = ready!(pinned.as_mut().poll(cx));
				if guard.counter > 0 {
					guard.counter -= 1;
					buf[..len].copy_from_slice(&u64::to_ne_bytes(1));
					wake_all(&mut guard.write_queue);
					Poll::Ready(Ok(len))
				} else {
					enqueue(&mut guard.read_queue, cx.waker());
					Poll::Pending
				}
			} else {
				let mut pinned = pin!(self.state.lock());
				let mut guard = ready!(pinned.as_mut().poll(cx));
				let tmp = guard.counter;
				if tmp > 0 {
					guard.counter = 0;
					buf[..len].copy_from_slice(&u64::to_ne_bytes(tmp));
					wake_all(&mut guard.write_queue);
					Poll::Ready(Ok(len))
				} else if self.flags.contains(EventFlags::EFD_NONBLOCK) {
					Poll::Ready(Err(Errno::Again))
				} else {
					enqueue(&mut guard.read_queue, cx.waker());
					Poll::Pending
				}
			}
		})
		.await
	}

	async fn write(&self, buf: &[u8]) -> io::Result<usize> {
		let len = mem::size_of::<u64>();

		if buf.len() < len {
			return Err(Errno::Inval);
		}

		let c = u64::from_ne_bytes(buf[..len].try_into().unwrap());

		future::poll_fn(|cx| {
			let mut pinned = pin!(self.state.lock());
			let mut guard = ready!(pinned.as_mut().poll(cx));
			if u64::MAX - guard.counter > c {
				guard.counter += c;
				wake_all(&mut guard.read_queue);

				Poll::Ready(Ok(len))
			} else if self.flags.contains(EventFlags::EFD_NONBLOCK) {
				Poll::Ready(Err(Errno::Again))
			} else {
				enqueue(&mut guard.write_queue, cx.waker());
				Poll::Pending
			}
		})
		.await
	}

	async fn poll(&self, event: PollEvent) -> io::Result<PollEvent> {
		let guard = self.state.lock().await;

		let mut available = PollEvent::empty();

		if guard.counter < u64::MAX - 1 {
			available.insert(PollEvent::POLLOUT | PollEvent::POLLWRNORM | PollEvent::POLLWRBAND);
		}

		if guard.counter > 0 {
			available.insert(PollEvent::POLLIN | PollEvent::POLLRDNORM | PollEvent::POLLRDBAND);
		}

		drop(guard);

		let ret = event & available;

		future::poll_fn(|cx| {
			if ret.is_empty() {
				let mut pinned = pin!(self.state.lock());
				let mut guard = ready!(pinned.as_mut().poll(cx));
				if event
					.intersects(PollEvent::POLLIN | PollEvent::POLLRDNORM | PollEvent::POLLRDNORM)
				{
					enqueue(&mut guard.read_queue, cx.waker());
					Poll::Pending
				} else if event
					.intersects(PollEvent::POLLOUT | PollEvent::POLLWRNORM | PollEvent::POLLWRBAND)
				{
					enqueue(&mut guard.write_queue, cx.waker());
					Poll::Pending
				} else {
					Poll::Ready(Ok(ret))
				}
			} else {
				Poll::Ready(Ok(ret))
			}
		})
		.await
	}
}
