use axum::{
    body::Body,
    extract::State,
    http::HeaderMap,
    response::Response,
};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;
use tracing::{error, info, warn};

use crate::error::ProxyError;
use crate::proxy::check_auth;
use crate::types::*;
use crate::AppState;

// ============================================================
// Request conversion: Anthropic → OpenAI
// ============================================================

fn convert_request(req: AnthropicRequest, model_map: &std::collections::HashMap<String, String>, max_tokens_cap: Option<u32>) -> OpenAIRequest {
    let mut messages: Vec<OpenAIMessage> = Vec::new();

    // Convert system prompt to a system message
    if let Some(system) = req.system {
        let text = match system {
            AnthropicSystem::Text(t) => t,
            AnthropicSystem::Blocks(blocks) => blocks
                .into_iter()
                .filter_map(|b| b.text)
                .collect::<Vec<_>>()
                .join("\n"),
        };
        if !text.is_empty() {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(OpenAIMessageContent::Text(text)),
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    // Convert messages
    for msg in req.messages {
        convert_message(msg, &mut messages);
    }

    // Convert tools — drop empty list so upstream doesn't reject `tools: []`
    let tools = req.tools.and_then(|tools| {
        if tools.is_empty() {
            return None;
        }
        Some(
            tools
                .into_iter()
                .map(|t| OpenAITool {
                    tool_type: "function".to_string(),
                    function: OpenAIFunction {
                        name: t.name,
                        description: t.description,
                        parameters: t.input_schema,
                    },
                })
                .collect(),
        )
    });

    // Convert tool_choice
    let tool_choice = req.tool_choice.map(|tc| convert_tool_choice(tc));

    let is_stream = req.stream.unwrap_or(false);

    OpenAIRequest {
        model: model_map
            .get(&req.model)
            .cloned()
            .unwrap_or(req.model),
        messages,
        max_tokens: Some(match max_tokens_cap {
            Some(cap) => req.max_tokens.min(cap),
            None => req.max_tokens,
        }),
        stream: req.stream,
        stream_options: if is_stream {
            Some(StreamOptions {
                include_usage: true,
            })
        } else {
            None
        },
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop_sequences,
        tools,
        tool_choice,
    }
}

fn convert_message(msg: AnthropicMessage, out: &mut Vec<OpenAIMessage>) {
    match msg.content {
        AnthropicContent::Text(text) => {
            out.push(OpenAIMessage {
                role: msg.role,
                content: Some(OpenAIMessageContent::Text(text)),
                tool_calls: None,
                tool_call_id: None,
            });
        }
        AnthropicContent::Blocks(blocks) => {
            if msg.role == "assistant" {
                convert_assistant_blocks(blocks, out);
            } else {
                // user message: may contain text, images, and tool_result
                convert_user_blocks(blocks, out);
            }
        }
    }
}

fn convert_assistant_blocks(blocks: Vec<AnthropicContentBlock>, out: &mut Vec<OpenAIMessage>) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<OpenAIToolCall> = Vec::new();

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text } => {
                text_parts.push(text);
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(OpenAIToolCall {
                    id: Some(id),
                    call_type: Some("function".to_string()),
                    function: OpenAIFunctionCall {
                        name: Some(name),
                        arguments: Some(serde_json::to_string(&input).unwrap_or_default()),
                    },
                    index: None,
                });
            }
            _ => {}
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(OpenAIMessageContent::Text(text_parts.join("\n")))
    };

    let tc = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    out.push(OpenAIMessage {
        role: "assistant".to_string(),
        content,
        tool_calls: tc,
        tool_call_id: None,
    });
}

fn convert_user_blocks(blocks: Vec<AnthropicContentBlock>, out: &mut Vec<OpenAIMessage>) {
    let mut content_parts: Vec<OpenAIContentPart> = Vec::new();
    let mut tool_results: Vec<(String, String)> = Vec::new(); // (tool_call_id, content)

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text } => {
                content_parts.push(OpenAIContentPart::Text { text });
            }
            AnthropicContentBlock::Image { source } => {
                // Convert Anthropic base64 image to OpenAI image_url format
                if let (Some(media_type), Some(data)) = (
                    source.get("media_type").and_then(|v| v.as_str()),
                    source.get("data").and_then(|v| v.as_str()),
                ) {
                    let url = format!("data:{};base64,{}", media_type, data);
                    content_parts.push(OpenAIContentPart::ImageUrl {
                        image_url: serde_json::json!({"url": url}),
                    });
                }
            }
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                let text = match content {
                    Some(ToolResultContent::Text(t)) => t,
                    Some(ToolResultContent::Blocks(blocks)) => blocks
                        .into_iter()
                        .filter_map(|b| b.text)
                        .collect::<Vec<_>>()
                        .join("\n"),
                    None => String::new(),
                };
                tool_results.push((tool_use_id, text));
            }
            _ => {}
        }
    }

    // Emit tool results as separate "tool" role messages (OpenAI format)
    for (tool_call_id, content) in tool_results {
        out.push(OpenAIMessage {
            role: "tool".to_string(),
            content: Some(OpenAIMessageContent::Text(content)),
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
        });
    }

    // Emit remaining content as a user message
    if !content_parts.is_empty() {
        let content = if content_parts.len() == 1 {
            if let OpenAIContentPart::Text { text } = &content_parts[0] {
                Some(OpenAIMessageContent::Text(text.clone()))
            } else {
                Some(OpenAIMessageContent::Parts(content_parts))
            }
        } else {
            Some(OpenAIMessageContent::Parts(content_parts))
        };
        out.push(OpenAIMessage {
            role: "user".to_string(),
            content,
            tool_calls: None,
            tool_call_id: None,
        });
    }
}

fn convert_tool_choice(tc: Value) -> Value {
    match &tc {
        Value::String(s) => match s.as_str() {
            "auto" => Value::String("auto".to_string()),
            "any" => Value::String("required".to_string()),
            "none" => Value::String("none".to_string()),
            _ => Value::String("auto".to_string()),
        },
        Value::Object(obj) => {
            if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
                serde_json::json!({
                    "type": "function",
                    "function": {"name": name}
                })
            } else {
                Value::String("auto".to_string())
            }
        }
        _ => Value::String("auto".to_string()),
    }
}

// ============================================================
// Response conversion: OpenAI → Anthropic (non-streaming)
// ============================================================

fn convert_response(resp: OpenAIResponse, model: &str) -> AnthropicResponse {
    let choice = resp.choices.into_iter().next();

    let mut content: Vec<AnthropicResponseContent> = Vec::new();
    let mut stop_reason = None;

    if let Some(c) = choice {
        // Convert finish_reason
        stop_reason = c.finish_reason.map(|r| match r.as_str() {
            "stop" => "end_turn".to_string(),
            "length" => "max_tokens".to_string(),
            "tool_calls" => "tool_use".to_string(),
            other => other.to_string(),
        });

        // Convert text content
        if let Some(text) = c.message.content {
            if !text.is_empty() {
                content.push(AnthropicResponseContent::Text { text });
            }
        }

        // Convert tool calls
        if let Some(tool_calls) = c.message.tool_calls {
            for tc in tool_calls {
                let input: Value = tc
                    .function
                    .arguments
                    .as_deref()
                    .and_then(|a| serde_json::from_str(a).ok())
                    .unwrap_or(Value::Object(Default::default()));
                content.push(AnthropicResponseContent::ToolUse {
                    id: tc.id.unwrap_or_else(|| format!("toolu_{}", generate_id())),
                    name: tc.function.name.unwrap_or_default(),
                    input,
                });
            }
        }
    }

    let usage = resp.usage.map(|u| AnthropicUsage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
    }).unwrap_or(AnthropicUsage {
        input_tokens: 0,
        output_tokens: 0,
    });

    AnthropicResponse {
        id: format!("msg_{}", generate_id()),
        resp_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: model.to_string(),
        stop_reason,
        stop_sequence: None,
        usage,
    }
}

// ============================================================
// Streaming conversion: OpenAI SSE → Anthropic SSE
// ============================================================

struct StreamState {
    msg_id: String,
    model: String,
    input_tokens: u32,
    output_tokens: u32,
    // Track which content blocks we've started
    current_text_index: Option<u32>,
    next_block_index: u32,
    // Accumulate tool call data (OpenAI streams them incrementally)
    tool_calls: std::collections::HashMap<u32, ToolCallAccum>,
    // Track if we've emitted message_start
    started: bool,
    buffer: String, // SSE line buffer for partial reads
}

struct ToolCallAccum {
    id: String,
    name: String,
    arguments: String,
    block_index: u32,
    started: bool,
}

impl StreamState {
    fn new(model: &str) -> Self {
        Self {
            msg_id: format!("msg_{}", generate_id()),
            model: model.to_string(),
            input_tokens: 0,
            output_tokens: 0,
            current_text_index: None,
            next_block_index: 0,
            tool_calls: std::collections::HashMap::new(),
            started: false,
            buffer: String::new(),
        }
    }

    fn message_start_event(&mut self) -> String {
        self.started = true;
        format_sse(
            "message_start",
            &serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": self.msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {
                        "input_tokens": self.input_tokens,
                        "output_tokens": 0
                    }
                }
            }),
        )
    }

    fn process_chunk(&mut self, chunk: &OpenAIChunk) -> String {
        let mut events = String::new();

        // Emit message_start on first chunk
        if !self.started {
            if let Some(usage) = &chunk.usage {
                self.input_tokens = usage.prompt_tokens;
            }
            events.push_str(&self.message_start_event());
            events.push_str(&format_sse("ping", &serde_json::json!({"type": "ping"})));
        }

        // Track usage from streaming chunks
        if let Some(usage) = &chunk.usage {
            self.input_tokens = usage.prompt_tokens;
            self.output_tokens = usage.completion_tokens;
        }

        for choice in &chunk.choices {
            let delta = match &choice.delta {
                Some(d) => d,
                None => continue,
            };

            // Handle text content delta
            if let Some(text) = &delta.content {
                if !text.is_empty() {
                    // Start text block if needed
                    if self.current_text_index.is_none() {
                        let idx = self.next_block_index;
                        self.next_block_index += 1;
                        self.current_text_index = Some(idx);
                        events.push_str(&format_sse(
                            "content_block_start",
                            &serde_json::json!({
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": {"type": "text", "text": ""}
                            }),
                        ));
                    }
                    events.push_str(&format_sse(
                        "content_block_delta",
                        &serde_json::json!({
                            "type": "content_block_delta",
                            "index": self.current_text_index.unwrap(),
                            "delta": {"type": "text_delta", "text": text}
                        }),
                    ));
                }
            }

            // Handle tool call deltas
            if let Some(tool_calls) = &delta.tool_calls {
                // Close text block before tool calls
                if let Some(idx) = self.current_text_index.take() {
                    events.push_str(&format_sse(
                        "content_block_stop",
                        &serde_json::json!({
                            "type": "content_block_stop",
                            "index": idx
                        }),
                    ));
                }

                for tc in tool_calls {
                    let tc_index = tc.index.unwrap_or(0);
                    let accum = self.tool_calls.entry(tc_index).or_insert_with(|| {
                        let block_index = self.next_block_index;
                        self.next_block_index += 1;
                        ToolCallAccum {
                            id: tc
                                .id
                                .clone()
                                .unwrap_or_else(|| format!("toolu_{}", generate_id())),
                            name: String::new(),
                            arguments: String::new(),
                            block_index,
                            started: false,
                        }
                    });

                    if let Some(name) = &tc.function.name {
                        accum.name.push_str(name);
                    }
                    if let Some(args) = &tc.function.arguments {
                        accum.arguments.push_str(args);
                    }
                    if let Some(id) = &tc.id {
                        if accum.id.is_empty() {
                            accum.id = id.clone();
                        }
                    }

                    // Emit content_block_start for this tool call
                    if !accum.started {
                        accum.started = true;
                        let block_idx = accum.block_index;
                        let id = accum.id.clone();
                        let name = accum.name.clone();
                        events.push_str(&format_sse(
                            "content_block_start",
                            &serde_json::json!({
                                "type": "content_block_start",
                                "index": block_idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": id,
                                    "name": name,
                                    "input": {}
                                }
                            }),
                        ));
                    }

                    // Emit argument delta
                    if let Some(args) = &tc.function.arguments {
                        if !args.is_empty() {
                            let block_idx = accum.block_index;
                            events.push_str(&format_sse(
                                "content_block_delta",
                                &serde_json::json!({
                                    "type": "content_block_delta",
                                    "index": block_idx,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": args
                                    }
                                }),
                            ));
                        }
                    }
                }
            }

            // Handle finish
            if let Some(reason) = &choice.finish_reason {
                let stop_reason = match reason.as_str() {
                    "stop" => "end_turn",
                    "length" => "max_tokens",
                    "tool_calls" => "tool_use",
                    other => other,
                };

                // Close any open text block
                if let Some(idx) = self.current_text_index.take() {
                    events.push_str(&format_sse(
                        "content_block_stop",
                        &serde_json::json!({
                            "type": "content_block_stop",
                            "index": idx
                        }),
                    ));
                }

                // Close any open tool call blocks
                let mut tool_indices: Vec<u32> = self
                    .tool_calls
                    .values()
                    .filter(|a| a.started)
                    .map(|a| a.block_index)
                    .collect();
                tool_indices.sort();
                for idx in tool_indices {
                    events.push_str(&format_sse(
                        "content_block_stop",
                        &serde_json::json!({
                            "type": "content_block_stop",
                            "index": idx
                        }),
                    ));
                }

                events.push_str(&format_sse(
                    "message_delta",
                    &serde_json::json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": stop_reason,
                            "stop_sequence": null
                        },
                        "usage": {
                            "output_tokens": self.output_tokens
                        }
                    }),
                ));

                events.push_str(&format_sse(
                    "message_stop",
                    &serde_json::json!({"type": "message_stop"}),
                ));
            }
        }

        events
    }

    /// Process raw SSE data from upstream, handling partial lines.
    fn process_raw_data(&mut self, data: &[u8]) -> String {
        let text = String::from_utf8_lossy(data);
        self.buffer.push_str(&text);

        let mut events = String::new();

        // Process complete lines
        while let Some(pos) = self.buffer.find('\n') {
            let line = self.buffer[..pos].trim().to_string();
            self.buffer = self.buffer[pos + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            if let Some(json_str) = line.strip_prefix("data: ") {
                if json_str.trim() == "[DONE]" {
                    // If we haven't emitted message_stop yet (no finish_reason in chunks)
                    continue;
                }
                match serde_json::from_str::<OpenAIChunk>(json_str) {
                    Ok(chunk) => {
                        events.push_str(&self.process_chunk(&chunk));
                    }
                    Err(e) => {
                        warn!("解析上游 SSE 数据失败: {} — raw: {}", e, json_str);
                    }
                }
            }
        }

        events
    }
}

fn format_sse(event: &str, data: &Value) -> String {
    format!("event: {}\ndata: {}\n\n", event, serde_json::to_string(data).unwrap())
}

fn generate_id() -> String {
    uuid::Uuid::new_v4()
        .to_string()
        .replace('-', "")
        .chars()
        .take(24)
        .collect()
}

// ============================================================
// Anthropic endpoint handler
// ============================================================

pub async fn anthropic_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ProxyError> {
    check_auth(&state, &headers)?;

    info!("收到 Anthropic 请求，body 大小: {} bytes", body.len());

    // Parse Anthropic request
    let anthropic_req: AnthropicRequest = serde_json::from_slice(&body).map_err(|e| {
        error!("解析 Anthropic 请求失败: {}", e);
        ProxyError::BadRequest(format!("Invalid request body: {}", e))
    })?;

    let is_stream = anthropic_req.stream.unwrap_or(false);
    let requested_model = anthropic_req.model.clone();

    // Convert to OpenAI format
    let openai_req = convert_request(
        anthropic_req,
        &state.model_map,
        state.max_tokens_cap,
    );
    let openai_body = serde_json::to_vec(&openai_req).map_err(|e| {
        ProxyError::InternalError(format!("Failed to serialize request: {}", e))
    })?;

    info!(
        "转换后 OpenAI 请求，model: {}, stream: {}",
        openai_req.model, is_stream
    );

    // Send to upstream
    let upstream_resp = state
        .client
        .post(state.upstream_url.as_str())
        .header("Content-Type", "application/json")
        .header(
            "Authorization",
            format!("Bearer {}", state.api_key.as_str()),
        )
        .body(openai_body)
        .send()
        .await
        .map_err(|e| {
            error!("上游请求失败: {}", e);
            ProxyError::UpstreamError(e.to_string())
        })?;

    let status = upstream_resp.status();
    info!("上游响应状态: {}", status);

    // If upstream returned an error, pass it through
    if !status.is_success() {
        let error_body = upstream_resp.text().await.unwrap_or_default();
        error!("上游返回错误: {}", error_body);
        return Err(ProxyError::UpstreamError(format!(
            "Upstream returned {}: {}",
            status, error_body
        )));
    }

    if is_stream {
        // Streaming: convert OpenAI SSE → Anthropic SSE
        let model = requested_model.clone();
        let mut stream_state = StreamState::new(&model);

        let byte_stream = upstream_resp.bytes_stream().map(move |chunk| {
            match chunk {
                Ok(data) => {
                    let events = stream_state.process_raw_data(&data);
                    Ok::<Bytes, std::io::Error>(Bytes::from(events))
                }
                Err(e) => {
                    error!("读取上游流数据失败: {}", e);
                    Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
                }
            }
        });

        let response = Response::builder()
            .status(200)
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .header("X-Accel-Buffering", "no")
            .body(Body::from_stream(byte_stream))
            .map_err(|e| ProxyError::InternalError(e.to_string()))?;

        Ok(response)
    } else {
        // Non-streaming: convert OpenAI JSON → Anthropic JSON
        let resp_body = upstream_resp.text().await.map_err(|e| {
            ProxyError::UpstreamError(format!("Failed to read upstream response: {}", e))
        })?;

        let openai_resp: OpenAIResponse = serde_json::from_str(&resp_body).map_err(|e| {
            error!("解析上游响应失败: {} — body: {}", e, resp_body);
            ProxyError::InternalError(format!("Failed to parse upstream response: {}", e))
        })?;

        let anthropic_resp = convert_response(openai_resp, &requested_model);
        let resp_json = serde_json::to_string(&anthropic_resp)
            .map_err(|e| ProxyError::InternalError(e.to_string()))?;

        let response = Response::builder()
            .status(200)
            .header("Content-Type", "application/json")
            .body(Body::from(resp_json))
            .map_err(|e| ProxyError::InternalError(e.to_string()))?;

        Ok(response)
    }
}
