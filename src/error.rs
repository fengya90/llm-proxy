use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

#[derive(Debug)]
pub enum ProxyError {
    Unauthorized,
    BadRequest(String),
    UpstreamError(String),
    InternalError(String),
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, message, error_type) = match self {
            ProxyError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "Invalid or missing API key".to_string(),
                "invalid_api_key",
            ),
            ProxyError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg, "invalid_request_error"),
            ProxyError::UpstreamError(msg) => (StatusCode::BAD_GATEWAY, msg, "proxy_error"),
            ProxyError::InternalError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, msg, "proxy_error")
            }
        };
        let body = serde_json::json!({
            "error": {
                "message": message,
                "type": error_type
            }
        });
        (status, axum::Json(body)).into_response()
    }
}
