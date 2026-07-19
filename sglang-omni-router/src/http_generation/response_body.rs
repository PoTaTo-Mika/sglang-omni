use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use http_body::{Frame, SizeHint};
use thiserror::Error;

use crate::worker_pool::RequestLease;

#[derive(Debug, Error)]
#[error("upstream response body terminated")]
pub(crate) struct RelayError;

pub(crate) struct DirectResponseBody {
    inner: Option<reqwest::Body>,
    lease: Option<RequestLease>,
    terminal: bool,
}

impl DirectResponseBody {
    pub(crate) fn new(inner: reqwest::Body, lease: RequestLease) -> Self {
        Self {
            inner: Some(inner),
            lease: Some(lease),
            terminal: false,
        }
    }

    fn terminalize(&mut self, upstream_failure: bool) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        if upstream_failure && let Some(lease) = self.lease.as_ref() {
            lease.request_immediate_probe();
        }
        drop(self.inner.take());
        drop(self.lease.take());
    }

    fn fail(&mut self) -> Poll<Option<Result<Frame<Bytes>, RelayError>>> {
        self.terminalize(true);
        Poll::Ready(Some(Err(RelayError)))
    }
}

impl Drop for DirectResponseBody {
    fn drop(&mut self) {
        self.terminalize(false);
    }
}

impl http_body::Body for DirectResponseBody {
    type Data = Bytes;
    type Error = RelayError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.terminal {
            return Poll::Ready(None);
        }
        let Some(inner) = self.inner.as_mut() else {
            return self.fail();
        };
        match Pin::new(inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => match frame.into_data() {
                Ok(data) => Poll::Ready(Some(Ok(Frame::data(data)))),
                Err(_trailers) => self.fail(),
            },
            Poll::Ready(Some(Err(_source))) => self.fail(),
            Poll::Ready(None) => {
                self.terminalize(false);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.terminal
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::collections::VecDeque;
    use std::future::poll_fn;
    use std::io;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::{Context, Poll, Waker};

    use bytes::Bytes;
    use http_body::{Body as _, Frame};
    use tokio::sync::Semaphore;

    use crate::config::RoutingStrategy;
    use crate::worker_pool::benchmark::Fixture;

    use super::DirectResponseBody;

    struct Frames(VecDeque<Result<Frame<Bytes>, io::Error>>);

    impl http_body::Body for Frames {
        type Data = Bytes;
        type Error = io::Error;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            Poll::Ready(self.0.pop_front())
        }
    }

    struct GatedFrames {
        ready: Arc<AtomicBool>,
        emitted: bool,
    }

    impl http_body::Body for GatedFrames {
        type Data = Bytes;
        type Error = io::Error;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            if !self.ready.load(Ordering::Acquire) {
                return Poll::Pending;
            }
            if self.emitted {
                Poll::Ready(None)
            } else {
                self.emitted = true;
                Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(b"late")))))
            }
        }
    }

    struct DropOrderFrames {
        frames: VecDeque<Result<Frame<Bytes>, io::Error>>,
        exact: Arc<Semaphore>,
        permits_when_dropped: Arc<AtomicUsize>,
    }

    impl http_body::Body for DropOrderFrames {
        type Data = Bytes;
        type Error = io::Error;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            Poll::Ready(self.frames.pop_front())
        }
    }

    impl Drop for DropOrderFrames {
        fn drop(&mut self) {
            self.permits_when_dropped
                .store(self.exact.available_permits(), Ordering::Release);
        }
    }

    struct CountingFrames {
        polls: Arc<AtomicUsize>,
        frames: VecDeque<Result<Frame<Bytes>, io::Error>>,
    }

    impl http_body::Body for CountingFrames {
        type Data = Bytes;
        type Error = io::Error;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            self.polls.fetch_add(1, Ordering::Relaxed);
            Poll::Ready(self.frames.pop_front())
        }
    }

    fn fixture() -> Fixture {
        Fixture::new(1, RoutingStrategy::RoundRobin).expect("response fixture must build")
    }

    #[tokio::test]
    async fn downstream_poll_is_the_only_upstream_poll_and_drop_releases_the_lease() {
        let fixture = fixture();
        let lease = fixture.request_lease().expect("acquire response lease");
        let polls = Arc::new(AtomicUsize::new(0));
        let upstream = reqwest::Body::wrap(CountingFrames {
            polls: Arc::clone(&polls),
            frames: VecDeque::from([
                Ok(Frame::data(Bytes::from_static(b"first"))),
                Ok(Frame::data(Bytes::from_static(b"second"))),
            ]),
        });
        let mut body = DirectResponseBody::new(upstream, lease);
        assert_eq!(polls.load(Ordering::Relaxed), 0);
        assert!(fixture.request_lease().is_none());

        let first = poll_fn(|cx| Pin::new(&mut body).poll_frame(cx))
            .await
            .expect("first frame")
            .expect("first frame succeeds")
            .into_data()
            .expect("first frame is data");
        assert_eq!(first, Bytes::from_static(b"first"));
        assert_eq!(polls.load(Ordering::Relaxed), 1);
        assert!(fixture.request_lease().is_none());

        drop(body);
        assert_eq!(polls.load(Ordering::Relaxed), 1);
        assert!(fixture.request_lease().is_some());
        assert!(!fixture.probe_requested());
    }

    #[tokio::test]
    async fn clean_eof_releases_lease_at_eof_without_waiting_for_wrapper_drop() {
        let fixture = fixture();
        let lease = fixture.request_lease().expect("acquire response lease");
        assert!(fixture.request_lease().is_none());
        let mut body = DirectResponseBody::new(reqwest::Body::from(Bytes::new()), lease);
        assert!(
            poll_fn(|cx| Pin::new(&mut body).poll_frame(cx))
                .await
                .is_none()
        );
        assert!(fixture.request_lease().is_some());
        assert!(!fixture.probe_requested());
    }

    #[tokio::test]
    async fn body_error_and_trailer_probe_then_release_at_the_terminal_poll() {
        for frame in [
            Err(io::Error::other("fixture")),
            Ok(Frame::trailers(axum::http::HeaderMap::new())),
        ] {
            let fixture = fixture();
            let lease = fixture.request_lease().expect("acquire response lease");
            let upstream = reqwest::Body::wrap(Frames(VecDeque::from([frame])));
            let mut body = DirectResponseBody::new(upstream, lease);
            let result = poll_fn(|cx| Pin::new(&mut body).poll_frame(cx)).await;
            assert!(result.is_some_and(|frame| frame.is_err()));
            assert!(fixture.probe_requested());
            assert!(fixture.request_lease().is_some());
        }
    }

    #[tokio::test]
    async fn no_postcommit_deadline_allows_a_later_pull_and_releases_at_eof() {
        let fixture = fixture();
        let lease = fixture.request_lease().expect("acquire response lease");
        let ready = Arc::new(AtomicBool::new(false));
        let mut body = DirectResponseBody::new(
            reqwest::Body::wrap(GatedFrames {
                ready: Arc::clone(&ready),
                emitted: false,
            }),
            lease,
        );
        let hint = body.size_hint();
        assert_eq!(hint.lower(), 0);
        assert_eq!(hint.upper(), None);
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        assert!(Pin::new(&mut body).poll_frame(&mut context).is_pending());
        assert!(fixture.request_lease().is_none());

        ready.store(true, Ordering::Release);
        let frame = match Pin::new(&mut body).poll_frame(&mut context) {
            Poll::Ready(Some(Ok(frame))) => Some(frame),
            Poll::Pending | Poll::Ready(None) | Poll::Ready(Some(Err(_))) => None,
        }
        .expect("gated data frame")
        .into_data()
        .expect("gated frame is data");
        assert_eq!(frame, Bytes::from_static(b"late"));
        assert!(matches!(
            Pin::new(&mut body).poll_frame(&mut context),
            Poll::Ready(None)
        ));
        assert!(!fixture.probe_requested());
        assert!(fixture.request_lease().is_some());
    }

    #[tokio::test]
    async fn every_terminal_path_drops_upstream_body_before_exact_lease() {
        let terminal_frames = [
            VecDeque::new(),
            VecDeque::from([Err(io::Error::other("fixture"))]),
            VecDeque::from([Ok(Frame::trailers(axum::http::HeaderMap::new()))]),
        ];
        for frames in terminal_frames {
            let fixture = fixture();
            let exact = fixture.exact_semaphore().expect("fixture exact semaphore");
            let lease = fixture.request_lease().expect("acquire response lease");
            let permits_when_dropped = Arc::new(AtomicUsize::new(usize::MAX));
            let upstream = reqwest::Body::wrap(DropOrderFrames {
                frames,
                exact,
                permits_when_dropped: Arc::clone(&permits_when_dropped),
            });
            let mut body = DirectResponseBody::new(upstream, lease);
            let _terminal = poll_fn(|cx| Pin::new(&mut body).poll_frame(cx)).await;
            assert_eq!(
                permits_when_dropped.load(Ordering::Acquire),
                0,
                "the upstream body observed a released exact lease"
            );
            assert!(fixture.request_lease().is_some());
        }

        let fixture = fixture();
        let exact = fixture.exact_semaphore().expect("fixture exact semaphore");
        let lease = fixture.request_lease().expect("acquire response lease");
        let permits_when_dropped = Arc::new(AtomicUsize::new(usize::MAX));
        let body = DirectResponseBody::new(
            reqwest::Body::wrap(DropOrderFrames {
                frames: VecDeque::from([Ok(Frame::data(Bytes::from_static(b"pending")))]),
                exact,
                permits_when_dropped: Arc::clone(&permits_when_dropped),
            }),
            lease,
        );
        drop(body);
        assert_eq!(permits_when_dropped.load(Ordering::Acquire), 0);
        assert!(fixture.request_lease().is_some());
        assert!(!fixture.probe_requested());
    }
}
