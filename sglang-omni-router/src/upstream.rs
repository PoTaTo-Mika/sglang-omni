use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::redirect::Policy;
use reqwest::{Client, Url};

use crate::config::{HttpGenerationConfig, WorkerConfig};
use crate::error::{ConfigError, RouterError};

/// One validated authority whose network destination cannot drift after startup.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedTarget {
    base_url: Url,
    hostname: Option<String>,
    socket_addr: SocketAddr,
}

impl ResolvedTarget {
    pub(crate) fn validate(worker: &WorkerConfig) -> Result<(), ConfigError> {
        Self::from_worker(worker).map(|_target| ())
    }

    pub(crate) fn from_worker(worker: &WorkerConfig) -> Result<Self, ConfigError> {
        let url = Url::parse(&worker.base_url).map_err(|_source| {
            ConfigError::invalid("workers.base_url", "must be a valid absolute URL")
        })?;
        if !matches!(url.scheme(), "http" | "https")
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || url.path() != "/"
            || url.port() == Some(0)
        {
            return Err(ConfigError::invalid(
                "workers.base_url",
                "must be an http(s) origin URL without credentials, query, fragment, or path",
            ));
        }
        let host = url
            .host_str()
            .ok_or_else(|| ConfigError::invalid("workers.base_url", "must contain a host"))?;
        let authority_ip = parse_ip_literal(host);
        let (hostname, transport_ip) = match (authority_ip, worker.resolved_ip) {
            (Some(authority), None) => (None, authority),
            (Some(authority), Some(resolved)) if authority == resolved => (None, authority),
            (None, Some(resolved))
                if host.is_ascii()
                    && !host.ends_with('.')
                    && !host.bytes().any(|byte| byte.is_ascii_uppercase()) =>
            {
                (Some(host.to_owned()), resolved)
            }
            (Some(_), Some(_)) => {
                return Err(ConfigError::invalid(
                    "workers.resolved_ip",
                    "must equal an IP-literal authority",
                ));
            }
            (None, None) => {
                return Err(ConfigError::invalid(
                    "workers.resolved_ip",
                    "is required for a DNS-name authority",
                ));
            }
            (None, Some(_)) => {
                return Err(ConfigError::invalid(
                    "workers.base_url",
                    "DNS authority must be lowercase ASCII without a trailing dot",
                ));
            }
        };
        let port = url.port_or_known_default().ok_or_else(|| {
            ConfigError::invalid("workers.base_url", "must have a known nonzero port")
        })?;
        Ok(Self {
            base_url: url,
            hostname,
            socket_addr: SocketAddr::new(transport_ip, port),
        })
    }

    pub(crate) fn endpoint(&self, path: &str) -> Url {
        let mut url = self.base_url.clone();
        url.set_path(path);
        url
    }

    fn resolver(&self) -> StaticResolver {
        let targets = self
            .hostname
            .as_ref()
            .map_or_else(HashMap::new, |hostname| {
                HashMap::from([(hostname.clone(), self.socket_addr)])
            });
        StaticResolver { targets }
    }
}

fn parse_ip_literal(host: &str) -> Option<IpAddr> {
    host.strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host)
        .parse()
        .ok()
}

struct StaticResolver {
    targets: HashMap<String, SocketAddr>,
}

#[derive(Debug)]
struct UnknownStaticTarget;

impl fmt::Display for UnknownStaticTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("worker target is not statically resolved")
    }
}

impl Error for UnknownStaticTarget {}

impl Resolve for StaticResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let result = self.targets.get(name.as_str()).copied().map_or_else(
            || Err(Box::new(UnknownStaticTarget) as Box<dyn Error + Send + Sync + 'static>),
            |address| {
                let address = SocketAddr::new(address.ip(), 0);
                let addresses: Addrs = Box::new(std::iter::once(address));
                Ok(addresses)
            },
        );
        Box::pin(std::future::ready(result))
    }
}

/// One immutable upstream transport owner shared by every request.
pub(crate) struct Upstream {
    pub(crate) target: ResolvedTarget,
    pub(crate) client: Client,
}

impl Upstream {
    pub(crate) fn build(
        worker: &WorkerConfig,
        transport: &HttpGenerationConfig,
    ) -> Result<Self, RouterError> {
        let target = ResolvedTarget::from_worker(worker)?;
        let client = Client::builder()
            .dns_resolver(Arc::new(target.resolver()))
            .no_proxy()
            .redirect(Policy::none())
            .retry(reqwest::retry::never())
            .http1_only()
            .connect_timeout(transport.connect_timeout())
            .pool_idle_timeout(Some(transport.pool_idle_timeout()))
            .pool_max_idle_per_host(transport.pool_max_idle_per_host)
            .no_gzip()
            .no_brotli()
            .no_zstd()
            .no_deflate()
            .build()
            .map_err(RouterError::GenerationClient)?;
        Ok(Self { target, client })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::net::{IpAddr, Ipv4Addr};
    use std::thread;
    use std::time::Duration;

    use crate::config::{HttpGenerationConfig, WorkerConfig};

    use super::{ResolvedTarget, Upstream};

    #[test]
    fn target_requires_an_exact_transport_identity() {
        let mut worker = WorkerConfig {
            worker_id: String::from("worker-a"),
            base_url: String::from("http://worker.invalid:8080/"),
            resolved_ip: None,
        };
        assert!(ResolvedTarget::from_worker(&worker).is_err());
        worker.resolved_ip = Some(IpAddr::V4(Ipv4Addr::LOCALHOST));
        let target = ResolvedTarget::from_worker(&worker).expect("pinned hostname target");
        assert_eq!(
            target.endpoint("/v1/chat/completions").host_str(),
            Some("worker.invalid")
        );
    }

    #[tokio::test]
    async fn pinned_hostname_connects_without_ambient_dns() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind pinned fixture");
        let address = listener.local_addr().expect("read fixture address");
        let server = thread::spawn(move || {
            let (mut stream, _peer) = listener.accept().expect("accept pinned request");
            stream
                .set_read_timeout(Some(Duration::from_secs(1)))
                .expect("bound fixture read");
            let mut request = [0_u8; 2048];
            let count = stream.read(&mut request).expect("read pinned request");
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                )
                .expect("write pinned response");
            String::from_utf8_lossy(&request[..count]).into_owned()
        });
        let worker = WorkerConfig {
            worker_id: String::from("worker-a"),
            base_url: format!("http://worker.invalid:{}/", address.port()),
            resolved_ip: Some(IpAddr::V4(Ipv4Addr::LOCALHOST)),
        };
        let transport = HttpGenerationConfig {
            streamed_request_max_bytes: 1024,
            connect_timeout_ms: 1000,
            request_timeout_ms: 1000,
            pool_idle_timeout_ms: 1000,
            pool_max_idle_per_host: 1,
        };
        let upstream = Upstream::build(&worker, &transport).expect("build pinned upstream");
        let response = upstream
            .client
            .post(upstream.target.endpoint("/v1/chat/completions"))
            .header("content-type", "application/json")
            .body("{}")
            .send()
            .await
            .expect("pinned request succeeds");
        assert_eq!(response.status(), 200);
        let request = server.join().expect("join pinned fixture");
        assert!(
            request
                .to_ascii_lowercase()
                .contains(&format!("host: worker.invalid:{}", address.port()))
        );
    }
}
