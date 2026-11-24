# Multi-stage build for tesserax.
# BACKEND selects the GPU feature and the runtime image: cuda or rocm.
# The CUDA image builds for ARM64 and x86_64, the ROCm image for x86_64.
ARG BACKEND=cuda

# The Rust version matches rust-toolchain.toml.
FROM rust:1.99.1-bookworm AS builder
ARG BACKEND

WORKDIR /src
COPY . .
# target/ stays in a build cache, so the layer holds only the binary.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --locked --release -p tessera-cli --bin tesserax --features "$BACKEND" \
    && cp target/release/tesserax /usr/local/bin/tesserax

# tesserax loads the CUDA driver and NVRTC at runtime. The driver comes
# from the host through the NVIDIA container runtime.
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04 AS runtime-cuda

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cuda-nvrtc-13-0 \
    && rm -rf /var/lib/apt/lists/*

# tesserax loads the HIP runtime and hiprtc at runtime. The full ROCm
# SDK is not necessary.
FROM ubuntu:24.04 AS runtime-rocm
ARG ROCM_VERSION=7.2

ADD --chmod=644 https://repo.radeon.com/rocm/rocm.gpg.key /etc/apt/keyrings/rocm.asc
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.asc] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
        > /etc/apt/sources.list.d/rocm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends hip-runtime-amd \
    && rm -rf /var/lib/apt/lists/* \
    && echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf \
    && ldconfig

FROM runtime-${BACKEND}

COPY --from=builder /usr/local/bin/tesserax /usr/local/bin/tesserax

EXPOSE 8080
ENTRYPOINT ["tesserax"]
