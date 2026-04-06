mod anthropic;
mod error;
mod proxy;
mod types;

use axum::{
    http::Method,
    routing::post,
    Router,
};
use clap::Parser;
use reqwest::Client;
use std::collections::HashMap;
use std::sync::Arc;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about = "LLM Proxy Server")]
struct Args {
    /// 监听地址
    #[arg(long, env = "PROXY_HOST", default_value = "0.0.0.0")]
    host: String,

    /// 监听端口
    #[arg(long, env = "PROXY_PORT", default_value_t = 8080)]
    port: u16,

    /// 服务的 base path（如 /v1）
    #[arg(long, env = "PROXY_BASE_PATH", default_value = "/v1")]
    base_path: String,

    /// 上游 LLM 服务地址
    #[arg(long, env = "UPSTREAM_URL")]
    upstream_url: String,

    /// 上游服务 API Key（Bearer token，注入给上游）
    #[arg(long, env = "API_KEY")]
    api_key: String,

    /// 客户端访问本代理所需的 API Key（支持多个，逗号分隔；不指定则不校验）
    #[arg(long, env = "PROXY_API_KEYS", value_delimiter = ',')]
    proxy_api_keys: Vec<String>,

    /// 模型映射表（格式: src1=dst1,src2=dst2；未匹配则透传）
    #[arg(long, env = "MODEL_MAP", value_delimiter = ',')]
    model_map: Vec<String>,

    /// Anthropic 转换时对 max_tokens 的上限约束（不指定则透传原始值）
    #[arg(long, env = "MAX_TOKENS_CAP")]
    max_tokens_cap: Option<u32>,
}

#[derive(Clone)]
pub struct AppState {
    pub client: Arc<Client>,
    pub upstream_url: Arc<String>,
    pub api_key: Arc<String>,
    pub proxy_api_keys: Arc<Vec<String>>,
    pub auth_enabled: bool,
    pub model_map: Arc<HashMap<String, String>>,
    pub max_tokens_cap: Option<u32>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "llm_proxy=info,tower_http=info".into()),
        )
        .init();

    let args = Args::parse();

    let listen: std::net::SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .expect("PROXY_HOST / PROXY_PORT 地址格式无效");

    let base = args.base_path.trim_end_matches('/');
    let openai_path = format!("{}/chat/completions", base);
    let anthropic_path = format!("{}/messages", base);

    let client = Client::builder()
        .build()
        .expect("Failed to build HTTP client");

    let auth_enabled = !args.proxy_api_keys.is_empty();

    let model_map: HashMap<String, String> = args
        .model_map
        .iter()
        .filter_map(|entry| {
            let (k, v) = entry.split_once('=')?;
            Some((k.trim().to_string(), v.trim().to_string()))
        })
        .collect();

    let state = AppState {
        client: Arc::new(client),
        upstream_url: Arc::new(args.upstream_url.clone()),
        api_key: Arc::new(args.api_key.clone()),
        proxy_api_keys: Arc::new(args.proxy_api_keys.clone()),
        auth_enabled,
        model_map: Arc::new(model_map),
        max_tokens_cap: args.max_tokens_cap,
    };

    let cors = CorsLayer::new()
        .allow_methods([Method::POST, Method::OPTIONS])
        .allow_headers(Any)
        .allow_origin(Any);

    info!("LLM Proxy 启动，监听 {}", listen);
    info!("  OpenAI    端点: {}{}", listen, openai_path);
    info!("  Anthropic 端点: {}{}", listen, anthropic_path);
    info!("上游地址: {}", args.upstream_url);
    if !state.model_map.is_empty() {
        info!("模型映射: {:?}", *state.model_map);
    }
    if let Some(cap) = args.max_tokens_cap {
        info!("max_tokens 上限: {}", cap);
    }
    if auth_enabled {
        info!("鉴权模式: 开启（proxy-api-keys 已配置）");
    } else {
        info!("鉴权模式: 关闭（无需客户端 API Key）");
    }

    let app = Router::new()
        .route(&openai_path, post(proxy::proxy_handler))
        .route(&anthropic_path, post(anthropic::anthropic_handler))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .expect("Failed to bind");

    axum::serve(listener, app)
        .await
        .expect("Server error");
}
