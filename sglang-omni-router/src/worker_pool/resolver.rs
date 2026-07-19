use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::error::Error;
use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::redirect::Policy;
use reqwest::{Client, Url};

use super::profile::WorkerConfig;

/// One immutable exact transport target derived only from validated input.
///
/// The URL authority remains authoritative for HTTP Host, certificate checks,
/// and SNI. The concrete socket is solely the pinned TCP destination.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedTarget {
    base_url: Url,
    health_url: Url,
    authority: CanonicalAuthority,
    transport_ip: IpAddr,
    socket_addr: SocketAddr,
}

#[derive(Clone, Debug)]
struct CanonicalAuthority {
    host: CanonicalHost,
    effective_port: u16,
}

#[derive(Clone, Debug)]
enum CanonicalHost {
    Domain(String),
    Ip(IpAddr),
}

impl ResolvedTarget {
    pub(super) fn from_worker(worker: &WorkerConfig) -> Option<Self> {
        Self::from_parts(
            worker.base_url.as_str(),
            worker.health_path.as_str(),
            worker.resolved_ip,
        )
    }

    pub(super) fn from_parts(
        base_url: &str,
        health_path: &str,
        resolved_ip: Option<IpAddr>,
    ) -> Option<Self> {
        let base_url = Url::parse(base_url).ok()?;
        if !matches!(base_url.scheme(), "http" | "https") || base_url.port() == Some(0) {
            return None;
        }
        let host = base_url.host_str()?;
        let (host, transport_ip) = match (parse_ip_literal(host), resolved_ip) {
            (Some(ip), None) => (CanonicalHost::Ip(ip), ip),
            (None, Some(ip))
                if host.is_ascii()
                    && !host.ends_with('.')
                    && !host.bytes().any(|byte| byte.is_ascii_uppercase()) =>
            {
                (CanonicalHost::Domain(host.to_owned()), ip)
            }
            _ => return None,
        };
        let effective_port = base_url.port_or_known_default()?;
        if effective_port == 0 {
            return None;
        }
        let mut health_url = base_url.clone();
        health_url.set_path(health_path);
        let socket_addr = SocketAddr::new(transport_ip, effective_port);
        Some(Self {
            base_url,
            health_url,
            authority: CanonicalAuthority {
                host,
                effective_port,
            },
            transport_ip,
            socket_addr,
        })
    }

    pub(crate) fn base_url(&self) -> &Url {
        &self.base_url
    }

    pub(super) fn health_url(&self) -> &Url {
        &self.health_url
    }

    pub(crate) fn transport_ip(&self) -> IpAddr {
        self.transport_ip
    }

    pub(crate) fn effective_port(&self) -> u16 {
        self.authority.effective_port
    }

    pub(crate) fn socket_addr(&self) -> SocketAddr {
        self.socket_addr
    }

    fn canonical_hostname(&self) -> Option<&str> {
        match &self.authority.host {
            CanonicalHost::Domain(host) => Some(host),
            CanonicalHost::Ip(_) => None,
        }
    }
}

fn parse_ip_literal(host: &str) -> Option<IpAddr> {
    host.strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host)
        .parse()
        .ok()
}

/// Immutable exact-name resolver with no fallback or asynchronous work.
/// Its map is bounded by the validated startup manifest.
pub(super) struct StaticResolver {
    targets: HashMap<String, SocketAddr>,
}

impl StaticResolver {
    pub(super) fn from_targets(targets: &[ResolvedTarget]) -> Option<Self> {
        let mut names = HashMap::with_capacity(targets.len());
        for target in targets {
            let Some(hostname) = target.canonical_hostname() else {
                continue;
            };
            let address = SocketAddr::new(target.transport_ip, 0);
            match names.entry(hostname.to_owned()) {
                Entry::Vacant(entry) => {
                    entry.insert(address);
                }
                Entry::Occupied(entry) if *entry.get() == address => {}
                Entry::Occupied(_) => return None,
            }
        }
        Some(Self { targets: names })
    }
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
                let addresses: Addrs = Box::new(std::iter::once(address));
                Ok(addresses)
            },
        );
        Box::pin(std::future::ready(result))
    }
}

pub(super) fn build_health_client(
    resolver: Arc<StaticResolver>,
    timeout: Duration,
    interval: Duration,
) -> Result<Client, reqwest::Error> {
    Client::builder()
        .dns_resolver(resolver)
        .no_proxy()
        .redirect(Policy::none())
        .retry(reqwest::retry::never())
        .http1_only()
        .connect_timeout(timeout)
        .timeout(timeout)
        .pool_idle_timeout(Some(interval.saturating_mul(2)))
        .pool_max_idle_per_host(1)
        .no_gzip()
        .no_brotli()
        .no_zstd()
        .no_deflate()
        .build()
}

pub(super) fn build_generation_client(
    resolver: Arc<StaticResolver>,
    connect_timeout: Duration,
    pool_idle_timeout: Duration,
    pool_max_idle_per_host: usize,
) -> Result<Client, reqwest::Error> {
    Client::builder()
        .dns_resolver(resolver)
        .no_proxy()
        .redirect(Policy::none())
        .retry(reqwest::retry::never())
        .http1_only()
        .connect_timeout(connect_timeout)
        .pool_idle_timeout(Some(pool_idle_timeout))
        .pool_max_idle_per_host(pool_max_idle_per_host)
        .no_gzip()
        .no_brotli()
        .no_zstd()
        .no_deflate()
        .build()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::future::Future;
    use std::io::{Read, Write};
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener};
    use std::str::FromStr;
    use std::task::{Context, Poll, Waker};
    use std::thread;
    use std::time::{Duration, Instant};

    use reqwest::dns::{Name, Resolve};

    use super::{ResolvedTarget, StaticResolver, build_health_client};

    #[test]
    fn resolver_hit_and_miss_are_immediately_ready_and_fail_closed() {
        let target = ResolvedTarget::from_parts(
            "http://worker-a.invalid:8443/",
            "/health",
            Some(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 10))),
        )
        .expect("valid pinned target");
        let resolver = StaticResolver::from_targets(&[target]).expect("valid static resolver");
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);

        let mut hit = resolver.resolve(Name::from_str("worker-a.invalid").expect("valid DNS name"));
        match Future::poll(hit.as_mut(), &mut context) {
            Poll::Ready(Ok(addresses)) => assert_eq!(
                addresses.collect::<Vec<_>>(),
                [SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 10)), 0)]
            ),
            _ => panic!("known static target must resolve immediately"),
        }

        let mut miss =
            resolver.resolve(Name::from_str("unknown.invalid").expect("valid missing DNS name"));
        match Future::poll(miss.as_mut(), &mut context) {
            Poll::Ready(Err(error)) => {
                assert_eq!(
                    error.to_string(),
                    "worker target is not statically resolved"
                );
                assert!(!error.to_string().contains("unknown"));
            }
            _ => panic!("unknown static target must fail immediately"),
        }
    }

    #[test]
    fn ip_literal_targets_derive_transport_ip_and_effective_port() {
        let ipv4 = ResolvedTarget::from_parts("https://127.0.0.1/", "/health", None)
            .expect("valid IPv4 target");
        assert_eq!(ipv4.transport_ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(ipv4.effective_port(), 443);
        assert_eq!(
            ipv4.socket_addr(),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 443)
        );

        let ipv6 = ResolvedTarget::from_parts("http://[::1]:18080/", "/health", None)
            .expect("valid IPv6 target");
        assert_eq!(ipv6.transport_ip(), IpAddr::V6(Ipv6Addr::LOCALHOST));
        assert_eq!(ipv6.effective_port(), 18_080);
        assert_eq!(
            ipv6.socket_addr(),
            SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 18_080)
        );
    }

    #[tokio::test]
    async fn pinned_reserved_hostname_connects_to_ip_and_preserves_host_authority() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind pinned-host fixture");
        listener
            .set_nonblocking(true)
            .expect("bound pinned-host accept");
        let address = listener.local_addr().expect("read pinned-host address");
        assert_ne!(address.port(), 80);
        let server = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(2);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && Instant::now() < deadline =>
                    {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(error) => panic!("accept pinned-host request: {error}"),
                }
            };
            stream
                .set_nonblocking(false)
                .expect("use blocking pinned-host stream");
            stream
                .set_read_timeout(Some(Duration::from_secs(1)))
                .expect("bound pinned-host read");
            let mut bytes = Vec::with_capacity(1024);
            let mut chunk = [0_u8; 512];
            while bytes.len() < 4096 && !bytes.windows(4).any(|part| part == b"\r\n\r\n") {
                let count = stream.read(&mut chunk).expect("read pinned-host request");
                if count == 0 {
                    break;
                }
                bytes.extend_from_slice(&chunk[..count]);
            }
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .expect("write pinned-host response");
            String::from_utf8(bytes).expect("pinned-host request must be ASCII")
        });
        let base_url = format!("http://branch-two-health.invalid:{}/", address.port());
        let target =
            ResolvedTarget::from_parts(&base_url, "/health", Some(IpAddr::V4(Ipv4Addr::LOCALHOST)))
                .expect("build pinned hostname target");
        let resolver = StaticResolver::from_targets(std::slice::from_ref(&target))
            .expect("build pinned hostname resolver");
        let client = build_health_client(
            std::sync::Arc::new(resolver),
            Duration::from_secs(1),
            Duration::from_secs(1),
        )
        .expect("build pinned hostname client");
        let response = client.get(target.health_url().clone()).send().await;
        let request = server.join().expect("join pinned-host fixture");
        let response = response.expect("pinned hostname request must connect without DNS");
        assert!(response.status().is_success());
        drop(response);
        let expected_host = format!("host: branch-two-health.invalid:{}", address.port());
        assert!(
            request
                .lines()
                .any(|line| line.eq_ignore_ascii_case(&expected_host))
        );
    }
}
