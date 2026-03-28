use axum::{
    body::Body,
    extract::State,
    http::HeaderMap,
    response::Response,
};
use bytes::Bytes;
use futures::StreamExt;
use tracing::{error, info};

use crate::error::ProxyError;
use crate::AppState;

/// Extract Bearer token from Authorization header.
pub fn extract_bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
}

/// Check proxy API key authentication.
pub fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), ProxyError> {
    if !state.auth_enabled {
        return Ok(());
    }
    // Support both Bearer token and x-api-key
    let token = extract_bearer_token(headers).or_else(|| {
        headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
    });
    let authorized = token
        .map(|t| state.proxy_api_keys.iter().any(|k| k == t))
        .unwrap_or(false);
    if !authorized {
        return Err(ProxyError::Unauthorized);
    }
    Ok(())
}

/// Transparent proxy handler for OpenAI-compatible requests.
pub async fn proxy_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ProxyError> {
    check_auth(&state, &headers)?;

    info!("收到 OpenAI 请求，body 大小: {} bytes", body.len());

    let mut req_builder = state
        .client
        .post(state.upstream_url.as_str())
        .header("Content-Type", "application/json")
        .header(
            "Authorization",
            format!("Bearer {}", state.api_key.as_str()),
        )
        .body(body);

    // Forward non-sensitive headers
    for (name, value) in &headers {
        let name_str = name.as_str().to_lowercase();
        if matches!(
            name_str.as_str(),
            "authorization"
                | "x-api-key"
                | "host"
                | "content-length"
                | "transfer-encoding"
                | "connection"
                | "content-type"
        ) {
            continue;
        }
        req_builder = req_builder.header(name, value);
    }

    let upstream_resp = req_builder.send().await.map_err(|e| {
        error!("上游请求失败: {}", e);
        ProxyError::UpstreamError(e.to_string())
    })?;

    let status = upstream_resp.status();
    info!("上游响应状态: {}", status);

    let content_type = upstream_resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let mut resp_builder = Response::builder().status(status.as_u16());

    for (name, value) in upstream_resp.headers() {
        let name_str = name.as_str().to_lowercase();
        if matches!(
            name_str.as_str(),
            "transfer-encoding" | "connection" | "keep-alive"
        ) {
            continue;
        }
        resp_builder = resp_builder.header(name, value);
    }

    if content_type.contains("text/event-stream") {
        resp_builder = resp_builder
            .header("Cache-Control", "no-cache")
            .header("X-Accel-Buffering", "no");
    }

    let byte_stream = upstream_resp.bytes_stream().map(|chunk| {
        chunk.map_err(|e| {
            error!("读取上游流数据失败: {}", e);
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })
    });

    let response = resp_builder
        .body(Body::from_stream(byte_stream))
        .map_err(|e| {
            error!("构建响应失败: {}", e);
            ProxyError::InternalError(e.to_string())
        })?;

    Ok(response)
}
