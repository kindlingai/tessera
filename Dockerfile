# Multi-stage CUDA build for tesserax (ARM64 / x86_64).
# The Rust version matches rust-toolchain.toml.
FROM rust:1.99.1-bookworm AS builder

WORKDIR /src
COPY . .
# target/ stays in a build cache, so the layer holds only the binary.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --locked --release -p tessera-cli --bin tesserax --features cuda \
    && cp target/release/tesserax /usr/local/bin/tesserax

FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cuda-nvrtc-13-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/tesserax /usr/local/bin/tesserax

EXPOSE 8080
ENTRYPOINT ["tesserax"]
