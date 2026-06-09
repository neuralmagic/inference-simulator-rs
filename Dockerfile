# A GPU-free model-server image for the llm-d P/D path: the real vLLM Rust frontend
# (vllm-rs) in front of our mock engine-core backend, with a minimal CPU-only libnixl
# (UCX backend) so the prefill/decode KV transfer runs over CPU RDMA / shared memory.
#
# Multi-stage: one Fedora builder compiles libnixl, our engine, and vllm-rs; the runtime
# stage carries just the libs + binaries.

ARG FEDORA_VERSION=42
# Keep NIXL_REF in lockstep with Cargo.toml's git dep so the data-plane ABI matches.
ARG NIXL_REF=41685d39
# The vllm-rs frontend source. Defaults to Will's fork at the lora-info-gauge commit, which is
# vLLM's ba94a3b plus the `vllm:lora_requests_info` exporter the upstream Rust frontend still
# lacks (see the divergence note). Our engine-core-client Cargo dep stays pinned at ba94a3b;
# the fork's wire protocol is compatible with it (verified by scripts/e2e_lora.sh). Once the
# gauge lands upstream, point these back at vllm-project/vllm @ ba94a3b.
ARG VLLM_REPO=https://github.com/wseaton/vllm.git
ARG VLLM_REF=e96171e4ecf911126295cee87173e7777b721484

# ---------------------------------------------------------------------------------------
FROM fedora:${FEDORA_VERSION} AS builder
ARG NIXL_REF
ARG VLLM_REPO
ARG VLLM_REF

# Toolchain: C/C++ + meson/ninja for libnixl, clang/libclang for bindgen (nixl-sys),
# ucx-devel for the UCX backend, perl for the vendored OpenSSL build (vllm-rs), protoc
# for vllm-rs's prost build.
RUN dnf install -y --setopt=install_weak_deps=False \
        gcc gcc-c++ make cmake meson ninja-build pkgconf-pkg-config git-core \
        autoconf automake libtool rdma-core-devel numactl-devel libstdc++-devel \
        clang clang-devel llvm-devel \
        protobuf-compiler perl-core python3 python3-devel pybind11-devel \
        curl ca-certificates findutils which xz \
    && dnf clean all

# Current stable Rust (edition 2024 needs >= 1.85; Fedora's may lag).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# 1. Build UCX from source. Fedora's packaged UCX is too old for current nixl
#    (missing UCS_BIT_GET etc.); the nixl contrib pins v1.21.x. CPU + RDMA, no CUDA.
RUN git clone --depth 1 -b v1.21.x https://github.com/openucx/ucx.git /src/ucx \
    && cd /src/ucx && ./autogen.sh \
    && ./configure --prefix=/usr/local/ucx --enable-mt \
        --without-cuda --without-rocm --with-verbs --with-rdmacm \
    && make -j"$(nproc)" && make install

# 2. Build a minimal, CPU-only libnixl (UCX plugin only; no CUDA/GDS, no tests/examples).
ENV PKG_CONFIG_PATH="/usr/local/ucx/lib/pkgconfig"
RUN git clone https://github.com/ai-dynamo/nixl.git /src/nixl \
    && cd /src/nixl && git checkout ${NIXL_REF} \
    && meson setup build --prefix=/usr/local --buildtype=release \
        -Denable_plugins=UCX \
        -Ddisable_gds_backend=true \
        -Dbuild_tests=false \
        -Dbuild_examples=false \
        -Drust=false \
        -Ducx_path=/usr/local/ucx \
    && ninja -C build install \
    && echo /usr/local/ucx/lib > /etc/ld.so.conf.d/nixl.conf \
    && echo /usr/local/lib64  >> /etc/ld.so.conf.d/nixl.conf \
    && echo /usr/local/lib    >> /etc/ld.so.conf.d/nixl.conf \
    && ldconfig
# Keep the system pkgconfig dirs so openssl (vllm-rs) is still found alongside ucx/nixl.
ENV PKG_CONFIG_PATH="/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig:/usr/local/ucx/lib/pkgconfig:/usr/lib64/pkgconfig:/usr/share/pkgconfig"

# vllm-rs needs openssl-sys (system variant) headers and the protobuf well-known types
# (its build.rs compiles a proto importing google/protobuf/struct.proto).
RUN dnf install -y --setopt=install_weak_deps=False openssl-devel protobuf-devel && dnf clean all
ENV PROTOC=/usr/bin/protoc
ENV PROTOC_INCLUDE=/usr/include

# 3. Build the vLLM Rust frontend (vllm-rs) from VLLM_REPO@VLLM_REF (see the ARG note above:
#    the fork carries the LoRA gauge; its protocol is compatible with our ba94a3b Cargo dep).
RUN git clone ${VLLM_REPO} /src/vllm \
    && cd /src/vllm && git checkout ${VLLM_REF} \
    && cd rust && cargo build --release --bin vllm-rs \
    && cp target/release/vllm-rs /usr/local/bin/vllm-rs

# 3. Build our mock engine against the real libnixl.
COPY . /src/inference-simulator-rs
RUN cd /src/inference-simulator-rs \
    && cargo build --release --features nixl \
    && cp target/release/inference-sim /usr/local/bin/inference-sim

# ---------------------------------------------------------------------------------------
FROM fedora:${FEDORA_VERSION} AS runtime
ARG FEDORA_VERSION

RUN dnf install -y --setopt=install_weak_deps=False \
        libstdc++ rdma-core numactl-libs ca-certificates bash \
    && dnf clean all

# Source-built UCX, libnixl + its UCX plugin, and the two binaries.
COPY --from=builder /usr/local/ucx/ /usr/local/ucx/
COPY --from=builder /usr/local/lib64/ /usr/local/lib64/
COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /usr/local/bin/vllm-rs /usr/local/bin/vllm-rs
COPY --from=builder /usr/local/bin/inference-sim /usr/local/bin/inference-sim
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN echo /usr/local/ucx/lib > /etc/ld.so.conf.d/nixl.conf \
    && echo /usr/local/lib64 >> /etc/ld.so.conf.d/nixl.conf \
    && echo /usr/local/lib >> /etc/ld.so.conf.d/nixl.conf \
    && ldconfig \
    && chmod +x /usr/local/bin/entrypoint.sh

# Tokenizer cache (vllm-rs fetches the tokenizer from HF at startup).
ENV HF_HOME=/tmp/hf
# modelserver (prefill) / vllm-behind-sidecar (decode) / NIXL metadata side channel /
# KV-cache events PUB (cache-aware routing).
EXPOSE 8000 8200 5600 5556

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
