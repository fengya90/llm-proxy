FROM scratch

COPY target/x86_64-unknown-linux-musl/release/llm-proxy /llm-proxy

EXPOSE 8080

ENTRYPOINT ["/llm-proxy"]
