use std::collections::HashSet;

use axum::http::header::{CONNECTION, HOST};
use axum::http::{HeaderMap, HeaderName, StatusCode};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, client_async_tls_with_config};

use crate::config::WebsocketConfig;
use crate::worker_pool::ResolvedTarget;

pub(super) type UpstreamSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ConnectError {
    InvalidRequest,
    ConnectTimeout,
    Connect,
    HandshakeTimeout,
    Handshake,
    Negotiation,
}

pub(super) async fn connect(
    target: &ResolvedTarget,
    path_and_query: &str,
    downstream_headers: &HeaderMap,
    policy: &WebsocketConfig,
) -> Result<UpstreamSocket, ConnectError> {
    let uri = target
        .websocket_uri(path_and_query)
        .ok_or(ConnectError::InvalidRequest)?;
    let mut request = uri
        .as_str()
        .into_client_request()
        .map_err(|_| ConnectError::InvalidRequest)?;
    copy_end_to_end_headers(downstream_headers, request.headers_mut())?;

    let tcp = tokio::time::timeout(
        policy.connect_timeout(),
        TcpStream::connect(target.socket_addr()),
    )
    .await
    .map_err(|_| ConnectError::ConnectTimeout)?
    .map_err(|_| ConnectError::Connect)?;
    tcp.set_nodelay(true).map_err(|_| ConnectError::Connect)?;
    let config = WebSocketConfig::default()
        .max_frame_size(Some(
            policy
                .frame_max_usize()
                .map_err(|_| ConnectError::InvalidRequest)?,
        ))
        .max_message_size(Some(
            policy
                .worker_message_max_usize()
                .map_err(|_| ConnectError::InvalidRequest)?,
        ));
    let (mut socket, response) = tokio::time::timeout(
        policy.handshake_timeout(),
        client_async_tls_with_config(request, tcp, Some(config), None),
    )
    .await
    .map_err(|_| ConnectError::HandshakeTimeout)?
    .map_err(|_| ConnectError::Handshake)?;
    if response.status() != StatusCode::SWITCHING_PROTOCOLS
        || response.headers().contains_key("sec-websocket-protocol")
        || response.headers().contains_key("sec-websocket-extensions")
    {
        let _ignored = socket.close(None).await;
        return Err(ConnectError::Negotiation);
    }
    Ok(socket)
}

pub(super) fn copy_end_to_end_headers(
    source: &HeaderMap,
    destination: &mut HeaderMap,
) -> Result<(), ConnectError> {
    let mut nominated = HashSet::new();
    for value in source.get_all(CONNECTION) {
        let text = value.to_str().map_err(|_| ConnectError::InvalidRequest)?;
        for token in text
            .split(',')
            .map(str::trim)
            .filter(|token| !token.is_empty())
        {
            let name = HeaderName::from_bytes(token.as_bytes())
                .map_err(|_| ConnectError::InvalidRequest)?;
            nominated.insert(name);
        }
    }
    for (name, value) in source {
        if name == HOST
            || is_hop_by_hop(name)
            || nominated.contains(name)
            || name.as_str().starts_with("sec-websocket-")
        {
            continue;
        }
        destination.append(name, value.clone());
    }
    Ok(())
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::result_large_err)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use axum::http::header::{AUTHORIZATION, CONNECTION, HOST};
    use axum::http::{HeaderMap, HeaderValue};
    use tokio::net::TcpListener;
    use tokio_tungstenite::accept_hdr_async;
    use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

    use crate::config::WebsocketConfig;
    use crate::worker_pool::ResolvedTarget;

    use super::{connect, copy_end_to_end_headers};

    #[test]
    fn strips_handshake_hop_and_connection_nominated_headers() {
        let mut source = HeaderMap::new();
        source.insert(HOST, HeaderValue::from_static("downstream.invalid"));
        source.insert(CONNECTION, HeaderValue::from_static("upgrade, x-private"));
        source.insert("upgrade", HeaderValue::from_static("websocket"));
        source.insert("x-private", HeaderValue::from_static("remove"));
        source.insert("sec-websocket-key", HeaderValue::from_static("remove"));
        source.insert(AUTHORIZATION, HeaderValue::from_static("Bearer retained"));
        let mut destination = HeaderMap::new();

        copy_end_to_end_headers(&source, &mut destination).expect("valid headers");

        assert_eq!(destination.len(), 1);
        assert_eq!(
            destination.get(AUTHORIZATION),
            Some(&HeaderValue::from_static("Bearer retained"))
        );
    }

    #[tokio::test]
    async fn pinned_ip_preserves_original_authority_query_and_headers() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind pinned websocket fixture");
        let address = listener.local_addr().expect("fixture address");
        let (observed_sender, observed_receiver) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept pinned websocket");
            let mut sender = Some(observed_sender);
            let mut socket =
                accept_hdr_async(stream, move |request: &Request, response: Response| {
                    let observed = (
                        request.uri().to_string(),
                        request
                            .headers()
                            .get(HOST)
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                        request
                            .headers()
                            .get(AUTHORIZATION)
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned),
                    );
                    if let Some(sender) = sender.take() {
                        let _sent = sender.send(observed);
                    }
                    Ok(response)
                })
                .await
                .expect("accept websocket handshake");
            let _closed = socket.close(None).await;
        });
        let target = ResolvedTarget::from_parts(
            &format!("http://branch-five.invalid:{}/", address.port()),
            "/health",
            Some(IpAddr::V4(Ipv4Addr::LOCALHOST)),
        )
        .expect("valid pinned hostname target");
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer retained"));

        let mut socket = connect(
            &target,
            "/v1/realtime?model=ignored",
            &headers,
            &WebsocketConfig::default(),
        )
        .await
        .expect("connect without ambient DNS");
        let observed = observed_receiver.await.expect("observe handshake");
        assert_eq!(observed.0, "/v1/realtime?model=ignored");
        assert_eq!(
            observed.1.as_deref(),
            Some(format!("branch-five.invalid:{}", address.port()).as_str())
        );
        assert_eq!(observed.2.as_deref(), Some("Bearer retained"));
        let _closed = socket.close(None).await;
        server.await.expect("join pinned websocket fixture");
    }
}
