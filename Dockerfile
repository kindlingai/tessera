# Multi-stage CUDA build for tesserax (ARM64 / x86_64).
FROM rust:1.99-bookworm AS builder

WORKDIR /src
COPY . .
RUN cargo build --release -p tessera-cli --bin tesserax --features cuda

FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cuda-nvrtc-13-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/tesserax /usr/local/bin/tesserax

EXPOSE 8080
ENTRYPOINT ["tesserax"]
