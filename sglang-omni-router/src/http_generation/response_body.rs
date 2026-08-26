use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use http_body::{Frame, SizeHint};
use thiserror::Error;

use crate::admission::AdmissionLease;

#[derive(Debug, Error)]
#[error("upstream response body terminated")]
pub(crate) struct RelayError;

/// Direct upstream response body whose terminal owner retains admission.
pub(crate) struct DirectResponseBody {
    inner: Option<reqwest::Body>,
    lease: Option<AdmissionLease>,
    terminal: bool,
}

impl DirectResponseBody {
    pub(crate) fn new(inner: reqwest::Body, lease: AdmissionLease) -> Self {
        Self {
            inner: Some(inner),
            lease: Some(lease),
            terminal: false,
        }
    }

    fn terminalize(&mut self) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        drop(self.inner.take());
        drop(self.lease.take());
    }

    fn fail(&mut self) -> Poll<Option<Result<Frame<Bytes>, RelayError>>> {
        self.terminalize();
        Poll::Ready(Some(Err(RelayError)))
    }
}

impl Drop for DirectResponseBody {
    fn drop(&mut self) {
        self.terminalize();
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
                self.terminalize();
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
