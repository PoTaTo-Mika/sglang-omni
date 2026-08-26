use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use axum::body::Body;
use bytes::Bytes;
use http_body::{Frame, SizeHint};
use sync_wrapper::SyncWrapper;
use thiserror::Error;
use tokio::time::Sleep;

use crate::error::HttpFault;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum UploadState {
    Incomplete,
    Failed(HttpFault),
    Complete,
}

pub(crate) type SharedUploadState = Arc<Mutex<UploadState>>;

#[derive(Debug, Error)]
#[error("request upload failed")]
pub(crate) struct UploadError;

pub(crate) struct DirectRequestBody {
    inner: SyncWrapper<Body>,
    expected: u64,
    maximum: u64,
    observed: u64,
    final_frame_returned: bool,
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
            final_frame_returned: false,
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
        if self.final_frame_returned {
            self.terminal = true;
            return Poll::Ready(None);
        }
        if tokio::time::Instant::now() >= self.deadline.deadline()
            || self.deadline.as_mut().poll(cx).is_ready()
        {
            return self.fail(HttpFault::RequestTimeout);
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
                        // Axum/Hyper owns fixed-length wire framing; do not hold this frame for a synthetic EOF.
                        if let Ok(mut state) = self.state.lock() {
                            *state = UploadState::Complete;
                        } else {
                            return self.fail(HttpFault::InternalError);
                        }
                        self.final_frame_returned = true;
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
        SizeHint::with_exact(self.expected.saturating_sub(self.observed))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::collections::VecDeque;
    use std::future::poll_fn;
    use std::io;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, Waker};
    use std::thread;
    use std::time::Duration;

    use super::{DirectRequestBody, HttpFault, UploadState};
    use axum::body::Body;
    use bytes::{Bytes, BytesMut};
    use http_body::{Body as _, Frame};

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
    async fn direct_upload_preserves_frames_and_completes_at_exact_length() {
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
    async fn direct_upload_skips_empty_data_and_completes_at_exact_length() {
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

        assert!(Pin::new(&mut direct).poll_frame(&mut context).is_pending());
        assert_eq!(polls.load(Ordering::Relaxed), 1);
        assert_eq!(
            *state.lock().expect("read cooperative upload state"),
            UploadState::Incomplete
        );
        let final_frame = match Pin::new(&mut direct).poll_frame(&mut context) {
            Poll::Ready(Some(Ok(frame))) => frame.into_data().ok(),
            Poll::Pending | Poll::Ready(None) | Poll::Ready(Some(Err(_))) => None,
        }
        .expect("exact declared length publishes the final data frame");
        assert_eq!(final_frame, Bytes::from_static(b"ab"));
        assert_eq!(polls.load(Ordering::Relaxed), 2);
        assert_eq!(
            *state.lock().expect("read completed upload state"),
            UploadState::Complete
        );
        assert!(matches!(
            Pin::new(&mut direct).poll_frame(&mut context),
            Poll::Ready(None)
        ));
        assert_eq!(polls.load(Ordering::Relaxed), 2);

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

    #[tokio::test]
    async fn reqwest_can_reuse_a_connection_after_the_direct_upload_terminal() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind pool fixture");
        let address = listener.local_addr().expect("read pool fixture address");
        let server = thread::spawn(move || {
            let (mut stream, _peer) = listener.accept().expect("accept pooled connection");
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("bound pooled read");
            for _ in 0..2 {
                let mut request = Vec::new();
                let mut chunk = [0_u8; 1024];
                while !request.windows(4).any(|part| part == b"\r\n\r\n") {
                    let count = stream.read(&mut chunk).expect("read pooled request head");
                    assert_ne!(count, 0, "pooled connection closed before next request");
                    request.extend_from_slice(&chunk[..count]);
                }
                while !request.ends_with(b"{}") {
                    let count = stream.read(&mut chunk).expect("read pooled request body");
                    assert_ne!(count, 0, "pooled request body closed early");
                    request.extend_from_slice(&chunk[..count]);
                }
                stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\n{}",
                    )
                    .expect("write pooled response");
            }
        });
        let client = reqwest::Client::builder()
            .http1_only()
            .build()
            .expect("build pool client");
        for _ in 0..2 {
            let state = Arc::new(Mutex::new(UploadState::Incomplete));
            let body = DirectRequestBody::new(
                Body::from("{}"),
                2,
                2,
                state,
                tokio::time::Instant::now() + Duration::from_secs(1),
            );
            let response = client
                .post(format!("http://{address}/v1/chat/completions"))
                .header("content-length", 2)
                .body(reqwest::Body::wrap(body))
                .send()
                .await
                .expect("send pooled direct body");
            assert_eq!(response.bytes().await.expect("consume pooled body"), "{}");
        }
        server.join().expect("join pooled fixture");
    }
}
