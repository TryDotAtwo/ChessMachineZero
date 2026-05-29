FROM gpu-dev-cutlass-nsight:2026-05-24

ENV DEBIAN_FRONTEND=noninteractive
ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_HOME=/opt/cargo
ENV PATH=/opt/cargo/bin:/usr/local/cuda/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        pkg-config \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable \
    && rustup component add rustfmt clippy \
    && rustc --version \
    && cargo --version \
    && cmake --version \
    && ninja --version \
    && nvcc --version \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

CMD ["bash", "-lc", "mkdir -p /work/test_results/native_container_logs && echo cmz-native-dev-ready >> /work/test_results/native_container_logs/container.log && echo cmz-native-dev-ready > /work/test_results/native_container_logs/docker-stream.log && tail -n +1 -F /work/test_results/native_container_logs/docker-stream.log"]
