# llm-proxy

一个轻量的 LLM 反向代理，用 Rust 编写。支持两种模式：

- **透传模式**：将 OpenAI 格式的请求原样转发给上游，替换鉴权头
- **转换模式**：将 Anthropic Messages API 格式的请求转换为 OpenAI Chat Completions 格式后转发

适用于将使用 Anthropic SDK 的客户端（如 Claude Code）接入任意 OpenAI-compatible 后端。

## 功能

- Anthropic Messages API → OpenAI Chat Completions 双向格式转换
  - 文本消息、图片、工具调用（tool use）、工具结果（tool result）
  - `system` 字段（字符串或 content block 数组）
  - `tool_choice`（`auto` / `any` / `none` / 指定函数名）
  - 流式（SSE）和非流式响应
  - `finish_reason` 双向映射（`stop`↔`end_turn`，`length`↔`max_tokens`，`tool_calls`↔`tool_use`）
- 客户端鉴权：支持多个 proxy API key（`Authorization: Bearer` 或 `x-api-key`）
- 模型名覆盖：强制将所有请求的 `model` 字段替换为指定值
- `max_tokens` 上限约束：可选，防止客户端传入超出上游限制的值
- 全请求头透传（过滤敏感头），流式响应逐块透传

## 构建

```bash
cargo build --release
# 产物：./target/release/llm-proxy
```

## 使用

```bash
llm-proxy \
  --upstream-url https://api.example.com/v1/chat/completions \
  --api-key sk-upstream-key
```

启动后自动暴露两个端点：

| 端点 | 说明 |
|------|------|
| `POST /{base-path}/chat/completions` | OpenAI 透传 |
| `POST /{base-path}/messages` | Anthropic 转换 |

默认 base path 为 `/v1`，即 `POST /v1/chat/completions` 和 `POST /v1/messages`。

## 参数

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| `--host` | `PROXY_HOST` | `0.0.0.0` | 监听地址 |
| `--port` | `PROXY_PORT` | `8080` | 监听端口 |
| `--base-path` | `PROXY_BASE_PATH` | `/v1` | 服务的 base path |
| `--upstream-url` | `UPSTREAM_URL` | 必填 | 上游 OpenAI-compatible 服务地址 |
| `--api-key` | `API_KEY` | 必填 | 注入给上游的 Bearer token |
| `--proxy-api-keys` | `PROXY_API_KEYS` | 无 | 客户端访问本代理所需的 key，逗号分隔；不指定则不校验 |
| `--model-override` | `MODEL_OVERRIDE` | 无 | 强制覆盖所有请求的模型名 |
| `--max-tokens-cap` | `MAX_TOKENS_CAP` | 无 | `max_tokens` 的最大值，超出则截断；不指定则透传原始值 |

## 示例

### 接入本地 Ollama

```bash
llm-proxy \
  --upstream-url http://localhost:11434/v1/chat/completions \
  --api-key ollama \
  --model-override qwen2.5:72b
```

### 接入 MiniMax 并限制输出长度

```bash
llm-proxy \
  --upstream-url https://api.minimax.chat/v1/text/chatcompletion_v2 \
  --api-key $MINIMAX_API_KEY \
  --model-override abab6.5s-chat \
  --max-tokens-cap 32768
```

### 启用客户端鉴权

```bash
llm-proxy \
  --upstream-url https://api.openai.com/v1/chat/completions \
  --api-key $OPENAI_API_KEY \
  --proxy-api-keys key1,key2,key3
```

客户端请求时携带任意一个配置的 key：

```
Authorization: Bearer key1
# 或
x-api-key: key1
```

### 配合 Claude Code 使用

将 `ANTHROPIC_BASE_URL` 指向本代理的 Anthropic 端点：

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080/v1
export ANTHROPIC_API_KEY=any-value  # 本代理不校验此值（除非配置了 --proxy-api-keys）
claude
```

## 日志

通过 `RUST_LOG` 环境变量控制日志级别：

```bash
RUST_LOG=debug llm-proxy ...
```

默认级别为 `info`。
