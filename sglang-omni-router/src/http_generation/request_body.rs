use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use axum::body::Body;
use bytes::Bytes;
use http_body::{Frame, SizeHint};
use sync_wrapper::SyncWrapper;
use thiserror::Error;
use tokio::sync::OwnedSemaphorePermit;
use tokio::time::Sleep;

use crate::error::HttpFault;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum UploadState {
    Incomplete,
    Failed(HttpFault),
    Complete,
}

pub(crate) type SharedUploadState = Arc<Mutex<UploadState>>;

pub(crate) struct BufferedBody {
    data: Option<Bytes>,
    _budget: OwnedSemaphorePermit,
}

impl BufferedBody {
    pub(crate) fn new(data: Bytes, budget: OwnedSemaphorePermit) -> Self {
        Self {
            data: Some(data),
            _budget: budget,
        }
    }
}

impl http_body::Body for BufferedBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Poll::Ready(self.data.take().map(|data| Ok(Frame::data(data))))
    }

    fn is_end_stream(&self) -> bool {
        self.data.is_none()
    }

    fn size_hint(&self) -> SizeHint {
        let remaining = self.data.as_ref().map_or(0, |data| data.len() as u64);
        SizeHint::with_exact(remaining)
    }
}

#[derive(Debug, Error)]
#[error("request upload failed")]
pub(crate) struct UploadError;

pub(crate) struct DirectRequestBody {
    inner: SyncWrapper<Body>,
    expected: u64,
    maximum: u64,
    observed: u64,
    final_frame: Option<Bytes>,
    eof_proven: bool,
    state: SharedUploadState,
    deadline: Pin<Box<Sleep>>,
    terminal: bool,
}

impl DirectRequestBody {
    pub(crate) fn new(
        body: Body,
        expected: u64,
        maximum: u64,
        state: SharedUploadState,
        deadline: tokio::time::Instant,
    ) -> Self {
        Self {
            inner: SyncWrapper::new(body),
            expected,
            maximum,
            observed: 0,
            final_frame: None,
            eof_proven: false,
            state,
            deadline: Box::pin(tokio::time::sleep_until(deadline)),
            terminal: false,
        }
    }

    fn fail(&mut self, fault: HttpFault) -> Poll<Option<Result<Frame<Bytes>, UploadError>>> {
        self.terminal = true;
        if let Ok(mut state) = self.state.lock() {
            *state = UploadState::Failed(fault);
        }
        Poll::Ready(Some(Err(UploadError)))
    }
}

impl http_body::Body for DirectRequestBody {
    type Data = Bytes;
    type Error = UploadError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.terminal {
            return Poll::Ready(None);
        }
        if self.eof_proven {
            self.terminal = true;
            return Poll::Ready(None);
        }
        if tokio::time::Instant::now() >= self.deadline.deadline()
            || self.deadline.as_mut().poll(cx).is_ready()
        {
            return self.fail(HttpFault::RequestTimeout);
        }
        if self.final_frame.is_some() {
            return match Pin::new(self.inner.get_mut()).poll_frame(cx) {
                Poll::Ready(None) => {
                    let Some(data) = self.final_frame.take() else {
                        return self.fail(HttpFault::InternalError);
                    };
                    self.eof_proven = true;
                    if let Ok(mut state) = self.state.lock() {
                        *state = UploadState::Complete;
                    } else {
                        return self.fail(HttpFault::InternalError);
                    }
                    Poll::Ready(Some(Ok(Frame::data(data))))
                }
                Poll::Ready(Some(Ok(frame))) => match frame.into_data() {
                    Ok(data) if data.is_empty() => {
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    }
                    Ok(_data) => self.fail(HttpFault::RequestBodyTooLarge),
                    Err(_trailers) => self.fail(HttpFault::MalformedRequest),
                },
                Poll::Ready(Some(Err(_source))) => self.fail(HttpFault::MalformedRequest),
                Poll::Pending => Poll::Pending,
            };
        }
        match Pin::new(self.inner.get_mut()).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => match frame.into_data() {
                Ok(data) => {
                    if data.is_empty() {
                        cx.waker().wake_by_ref();
                        return Poll::Pending;
                    }
                    let Ok(length) = u64::try_from(data.len()) else {
                        return self.fail(HttpFault::RequestBodyTooLarge);
                    };
                    let Some(observed) = self.observed.checked_add(length) else {
                        return self.fail(HttpFault::RequestBodyTooLarge);
                    };
                    if observed > self.expected || observed > self.maximum {
                        return self.fail(HttpFault::RequestBodyTooLarge);
                    }
                    self.observed = observed;
                    if observed == self.expected {
                        self.final_frame = Some(data);
                        cx.waker().wake_by_ref();
                        return Poll::Pending;
                    }
                    Poll::Ready(Some(Ok(Frame::data(data))))
                }
                Err(_trailers) => self.fail(HttpFault::MalformedRequest),
            },
            Poll::Ready(Some(Err(_source))) => self.fail(HttpFault::MalformedRequest),
            Poll::Ready(None) => {
                self.terminal = true;
                if self.observed != self.expected {
                    return self.fail(HttpFault::MalformedRequest);
                }
                if let Ok(mut state) = self.state.lock() {
                    *state = UploadState::Complete;
                } else {
                    return self.fail(HttpFault::InternalError);
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.terminal
    }

    fn size_hint(&self) -> SizeHint {
        // Reqwest receives the canonical Content-Length explicitly. Keeping
        // this wrapper's hint unknown forces Hyper to poll clean EOF instead
        // of stopping after the last declared data byte, which is required
        // before shared upload state may transition to Complete.
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, Waker};
    use std::time::Duration;

    use axum::body::Body;
    use bytes::{Bytes, BytesMut};
    use http_body::{Body as _, Frame};
    use tokio::sync::Semaphore;

    use super::{BufferedBody, DirectRequestBody, HttpFault, UploadState};

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

    struct Pending;

    impl http_body::Body for Pending {
        type Data = Bytes;
        type Error = io::Error;

        fn poll_frame(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            Poll::Pending
        }
    }

    struct AlwaysReady;

    impl http_body::Body for AlwaysReady {
        type Data = Bytes;
        type Error = io::Error;

        fn poll_frame(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(b"late")))))
        }
    }

    struct CountedFrames {
        frames: VecDeque<Result<Frame<Bytes>, io::Error>>,
        polls: Arc<AtomicUsize>,
    }

    impl http_body::Body for CountedFrames {
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

    async fn drive(body: Body, expected: u64) -> (Bytes, UploadState) {
        let state = Arc::new(Mutex::new(UploadState::Incomplete));
        let mut direct = DirectRequestBody::new(
            body,
            expected,
            64,
            Arc::clone(&state),
            tokio::time::Instant::now() + Duration::from_secs(1),
        );
        let mut output = BytesMut::new();
        while let Some(Ok(frame)) = poll_fn(|cx| Pin::new(&mut direct).poll_frame(cx)).await {
            output.extend_from_slice(
                &frame
                    .into_data()
                    .expect("direct body only returns data frames"),
            );
        }
        let terminal = *state.lock().expect("read upload state");
        (output.freeze(), terminal)
    }

    #[tokio::test]
    async fn direct_upload_preserves_frames_and_completes_only_at_exact_eof() {
        let frames = Frames(VecDeque::from([
            Ok(Frame::data(Bytes::from_static(b"ab"))),
            Ok(Frame::data(Bytes::from_static(b"c"))),
        ]));
        let (output, state) = drive(Body::new(frames), 3).await;
        assert_eq!(output, Bytes::from_static(b"abc"));
        assert_eq!(state, UploadState::Complete);

        let (_, short) = drive(Body::from("ab"), 3).await;
        assert_eq!(short, UploadState::Failed(HttpFault::MalformedRequest));
        let (_, long) = drive(Body::from("abcd"), 3).await;
        assert_eq!(long, UploadState::Failed(HttpFault::RequestBodyTooLarge));
    }

    #[tokio::test]
    async fn direct_upload_rejects_trailers_body_errors_and_deadlines() {
        let trailers = Frames(VecDeque::from([Ok(Frame::trailers(
            axum::http::HeaderMap::new(),
        ))]));
        let (_, trailer_state) = drive(Body::new(trailers), 0).await;
        assert_eq!(
            trailer_state,
            UploadState::Failed(HttpFault::MalformedRequest)
        );
        let errors = Frames(VecDeque::from([Err(io::Error::other("fixture"))]));
        let (_, error_state) = drive(Body::new(errors), 0).await;
        assert_eq!(
            error_state,
            UploadState::Failed(HttpFault::MalformedRequest)
        );

        let state = Arc::new(Mutex::new(UploadState::Incomplete));
        let mut direct = DirectRequestBody::new(
            Body::new(Pending),
            1,
            1,
            Arc::clone(&state),
            tokio::time::Instant::now(),
        );
        let result = poll_fn(|cx| Pin::new(&mut direct).poll_frame(cx)).await;
        assert!(result.is_some_and(|frame| frame.is_err()));
        assert_eq!(
            *state.lock().expect("read deadline state"),
            UploadState::Failed(HttpFault::RequestTimeout)
        );

        let ready_state = Arc::new(Mutex::new(UploadState::Incomplete));
        let mut ready = DirectRequestBody::new(
            Body::new(AlwaysReady),
            4,
            64,
            Arc::clone(&ready_state),
            tokio::time::Instant::now(),
        );
        let result = poll_fn(|cx| Pin::new(&mut ready).poll_frame(cx)).await;
        assert!(result.is_some_and(|frame| frame.is_err()));
        assert_eq!(
            *ready_state
                .lock()
                .expect("read authoritative deadline state"),
            UploadState::Failed(HttpFault::RequestTimeout)
        );
    }

    #[tokio::test]
    async fn direct_upload_skips_empty_data_cooperatively_through_exact_eof_proof() {
        let polls = Arc::new(AtomicUsize::new(0));
        let state = Arc::new(Mutex::new(UploadState::Incomplete));
        let mut direct = DirectRequestBody::new(
            Body::new(CountedFrames {
                frames: VecDeque::from([
                    Ok(Frame::data(Bytes::new())),
                    Ok(Frame::data(Bytes::from_static(b"ab"))),
                    Ok(Frame::data(Bytes::new())),
                    Ok(Frame::data(Bytes::new())),
                ]),
                polls: Arc::clone(&polls),
            }),
            2,
            64,
            Arc::clone(&state),
            tokio::time::Instant::now() + Duration::from_secs(1),
        );
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);

        for expected_polls in 1..=4 {
            assert!(Pin::new(&mut direct).poll_frame(&mut context).is_pending());
            assert_eq!(polls.load(Ordering::Relaxed), expected_polls);
            assert_eq!(
                *state.lock().expect("read cooperative upload state"),
                UploadState::Incomplete
            );
        }
        let final_frame = match Pin::new(&mut direct).poll_frame(&mut context) {
            Poll::Ready(Some(Ok(frame))) => frame.into_data().ok(),
            Poll::Pending | Poll::Ready(None) | Poll::Ready(Some(Err(_))) => None,
        }
        .expect("exact EOF must publish the retained final data frame");
        assert_eq!(final_frame, Bytes::from_static(b"ab"));
        assert_eq!(polls.load(Ordering::Relaxed), 5);
        assert_eq!(
            *state.lock().expect("read completed upload state"),
            UploadState::Complete
        );
        assert!(matches!(
            Pin::new(&mut direct).poll_frame(&mut context),
            Poll::Ready(None)
        ));
        assert_eq!(polls.load(Ordering::Relaxed), 5);

        let empty_frames = Frames(VecDeque::from([
            Ok(Frame::data(Bytes::new())),
            Ok(Frame::data(Bytes::new())),
        ]));
        let empty_state = Arc::new(Mutex::new(UploadState::Incomplete));
        let mut empty = DirectRequestBody::new(
            Body::new(empty_frames),
            0,
            64,
            Arc::clone(&empty_state),
            tokio::time::Instant::now() + Duration::from_secs(1),
        );
        assert!(Pin::new(&mut empty).poll_frame(&mut context).is_pending());
        assert!(Pin::new(&mut empty).poll_frame(&mut context).is_pending());
        assert!(matches!(
            Pin::new(&mut empty).poll_frame(&mut context),
            Poll::Ready(None)
        ));
        assert_eq!(
            *empty_state.lock().expect("read zero-length upload state"),
            UploadState::Complete
        );
    }

    #[test]
    fn buffered_bytes_own_the_aggregate_permit_until_consumed_or_dropped() {
        let semaphore = Arc::new(Semaphore::new(4));
        let permit = Arc::clone(&semaphore)
            .try_acquire_many_owned(3)
            .expect("reserve buffered bytes");
        let body = BufferedBody::new(Bytes::from_static(b"abc"), permit);
        assert_eq!(semaphore.available_permits(), 1);
        drop(body);
        assert_eq!(semaphore.available_permits(), 4);
    }
}
