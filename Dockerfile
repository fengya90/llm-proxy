FROM alpine

RUN apk add --no-cache tzdata ca-certificates \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone


EXPOSE 8080

ENTRYPOINT ["/llm-proxy"]
