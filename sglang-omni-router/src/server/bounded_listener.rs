use std::future::pending;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::serve::Listener;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::trace;

/// TCP listener that bounds sockets accepted into Axum connection tasks.
pub(super) struct BoundedTcpListener {
    listener: TcpListener,
    permits: Arc<Semaphore>,
}

impl BoundedTcpListener {
    /// Wraps a bound TCP listener with `max_connections` permits.
    pub(super) fn new(listener: TcpListener, max_connections: usize) -> Self {
        Self {
            listener,
            permits: Arc::new(Semaphore::new(max_connections)),
        }
    }
}

impl Listener for BoundedTcpListener {
    type Io = ConnectionIo;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        let permit = match Arc::clone(&self.permits).acquire_owned().await {
            Ok(permit) => permit,
            Err(_closed) => pending().await,
        };
        let (stream, address) = Listener::accept(&mut self.listener).await;
        if let Err(error) = stream.set_nodelay(true) {
            trace!(%error, "failed to enable TCP_NODELAY on client connection");
        }
        (ConnectionIo::new(stream, permit), address)
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}

/// Accepted socket whose lifetime owns one listener-capacity permit.
pub(super) struct ConnectionIo {
    stream: TcpStream,
    _permit: OwnedSemaphorePermit,
}

impl ConnectionIo {
    fn new(stream: TcpStream, permit: OwnedSemaphorePermit) -> Self {
        Self {
            stream,
            _permit: permit,
        }
    }
}

impl AsyncRead for ConnectionIo {
    fn poll_read(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_read(context, buffer)
    }
}

impl AsyncWrite for ConnectionIo {
    fn poll_write(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        Pin::new(&mut self.stream).poll_write(context, buffer)
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        Pin::new(&mut self.stream).poll_flush(context)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        Pin::new(&mut self.stream).poll_shutdown(context)
    }

    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffers: &[io::IoSlice<'_>],
    ) -> Poll<Result<usize, io::Error>> {
        Pin::new(&mut self.stream).poll_write_vectored(context, buffers)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

    use std::io;
    use std::sync::Arc;
    use std::time::Duration;

    use axum::Router;
    use axum::routing::get;
    use axum::serve::Listener as _;
    use tokio::net::TcpStream;

    use super::BoundedTcpListener;

    async fn listener(capacity: usize) -> (BoundedTcpListener, std::net::SocketAddr) {
        let tcp = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind isolated listener");
        let address = tcp.local_addr().expect("read isolated listener address");
        (BoundedTcpListener::new(tcp, capacity), address)
    }

    async fn write_all(stream: &TcpStream, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            stream
                .writable()
                .await
                .expect("wait for writable test socket");
            match stream.try_write(bytes) {
                Ok(written) => bytes = &bytes[written..],
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                Err(error) => panic!("write test socket: {error}"),
            }
        }
    }

    async fn valid_request(address: std::net::SocketAddr) {
        let stream = TcpStream::connect(address)
            .await
            .expect("connect valid client");
        write_all(
            &stream,
            b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        .await;
        let response = tokio::time::timeout(Duration::from_secs(1), async {
            let mut response = Vec::new();
            loop {
                stream
                    .readable()
                    .await
                    .expect("wait for readable test socket");
                let mut buffer = [0_u8; 256];
                match stream.try_read(&mut buffer) {
                    Ok(0) => break,
                    Ok(read) => response.extend_from_slice(&buffer[..read]),
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                    Err(error) => panic!("read test socket: {error}"),
                }
            }
            response
        })
        .await
        .expect("valid request should complete before deadline");
        assert!(response.starts_with(b"HTTP/1.1 200"));
        assert!(response.ends_with(b"ok"));
    }

    #[tokio::test]
    async fn bounds_accepted_wrappers_and_wakes_after_drop() {
        let (mut listener, address) = listener(1).await;
        let _first_client = TcpStream::connect(address)
            .await
            .expect("connect first client");
        let (first_io, _) = listener.accept().await;
        assert_eq!(listener.permits.available_permits(), 0);

        let _second_client = TcpStream::connect(address)
            .await
            .expect("connect queued client");
        assert!(
            tokio::time::timeout(Duration::from_millis(50), listener.accept())
                .await
                .is_err(),
            "second wrapper must wait while the only permit is owned"
        );

        drop(first_io);
        let (second_io, _) = tokio::time::timeout(Duration::from_secs(1), listener.accept())
            .await
            .expect("waiting accept should wake after wrapper drop");
        assert_eq!(listener.permits.available_permits(), 0);
        drop(second_io);
        assert_eq!(listener.permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn accepted_sockets_enable_tcp_nodelay() {
        let (mut listener, address) = listener(1).await;
        let _client = TcpStream::connect(address)
            .await
            .expect("connect isolated client");

        let (connection, _) = listener.accept().await;

        assert!(
            connection.stream.nodelay().expect("read TCP_NODELAY"),
            "accepted client sockets must disable Nagle's algorithm"
        );
    }

    #[tokio::test]
    async fn cancelled_accept_returns_its_pre_accept_permit() {
        let (mut listener, _address) = listener(1).await;

        assert!(
            tokio::time::timeout(Duration::from_millis(50), listener.accept())
                .await
                .is_err(),
            "accept without a client should remain pending"
        );
        assert_eq!(listener.permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn wrapper_drop_is_exact_once_and_nonblocking() {
        let (mut listener, address) = listener(1).await;
        let _client = TcpStream::connect(address)
            .await
            .expect("connect isolated client");
        let (io, _) = listener.accept().await;
        let permits = Arc::clone(&listener.permits);

        tokio::time::timeout(Duration::from_millis(50), async move { drop(io) })
            .await
            .expect("wrapper drop must not block");
        let permit = Arc::clone(&permits)
            .try_acquire_owned()
            .expect("drop must return exactly one permit");
        assert!(Arc::clone(&permits).try_acquire_owned().is_err());
        drop(permit);
        assert_eq!(permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn graceful_shutdown_retains_permit_until_confirmed_handler_finishes() {
        let (listener, address) = listener(1).await;
        let permits = Arc::clone(&listener.permits);
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let app = Router::new().route(
            "/",
            get({
                let entered = Arc::clone(&entered);
                let release = Arc::clone(&release);
                move || {
                    let entered = Arc::clone(&entered);
                    let release = Arc::clone(&release);
                    async move {
                        entered.notify_one();
                        release.notified().await;
                        "ok"
                    }
                }
            }),
        );
        let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel::<()>();
        let mut server = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _result = shutdown_receiver.await;
                })
                .await
        });

        let stream = TcpStream::connect(address)
            .await
            .expect("connect confirmed active handler");
        write_all(
            &stream,
            b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        .await;
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .expect("handler should confirm application entry");
        assert_eq!(permits.available_permits(), 0);

        shutdown_sender
            .send(())
            .expect("notify graceful test shutdown");
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut server)
                .await
                .is_err(),
            "graceful shutdown must retain an active handler"
        );
        assert_eq!(permits.available_permits(), 0);

        release.notify_one();
        let response = tokio::time::timeout(Duration::from_secs(1), async {
            let mut response = Vec::new();
            loop {
                stream.readable().await.expect("wait for graceful response");
                let mut buffer = [0_u8; 256];
                match stream.try_read(&mut buffer) {
                    Ok(0) => break,
                    Ok(read) => response.extend_from_slice(&buffer[..read]),
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                    Err(error) => panic!("read graceful response: {error}"),
                }
            }
            response
        })
        .await
        .expect("graceful response should complete before deadline");
        assert!(response.starts_with(b"HTTP/1.1 200"));
        assert!(response.ends_with(b"ok"));
        server
            .await
            .expect("graceful test server task should join")
            .expect("graceful test server should stop cleanly");
        assert_eq!(permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn axum_releases_permits_after_eof_malformed_input_and_disconnect() {
        let (listener, address) = listener(1).await;
        let permits = Arc::clone(&listener.permits);
        let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel::<()>();
        let mut server = tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/", get(|| async { "ok" })))
                .with_graceful_shutdown(async move {
                    let _result = shutdown_receiver.await;
                })
                .await
        });

        let eof = TcpStream::connect(address)
            .await
            .expect("connect EOF client");
        drop(eof);
        valid_request(address).await;

        let malformed = TcpStream::connect(address)
            .await
            .expect("connect malformed client");
        write_all(&malformed, b"not-http\r\n\r\n").await;
        valid_request(address).await;

        let disconnected = TcpStream::connect(address)
            .await
            .expect("connect disconnecting client");
        write_all(&disconnected, b"GET / HTTP/1.1\r\nHost: localhost\r\n").await;
        drop(disconnected);
        valid_request(address).await;
        drop(malformed);

        shutdown_sender
            .send(())
            .expect("notify test server shutdown");
        tokio::time::timeout(Duration::from_secs(1), &mut server)
            .await
            .expect("test server should stop before deadline")
            .expect("test server task should join")
            .expect("test server should stop cleanly");
        assert_eq!(permits.available_permits(), 1);
    }
}
