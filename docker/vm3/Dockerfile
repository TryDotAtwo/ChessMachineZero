FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gcc g++ cmake ninja-build python3 python3-pip \
      libssl-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN curl -fsSLo torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
      'https://download.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl' \
    && echo 'd6c3cba198dc93f93422a8545f48a6697890366e4b9701f54351fc27e2304bd3  torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl' | sha256sum -c - \
    && python3 -m pip install --no-deps ./torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
    && rm torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
    && mkdir -p /opt/pytorch \
    && ln -s /usr/local/lib/python3.10/dist-packages/torch/include /opt/pytorch/include \
    && ln -s /usr/local/lib/python3.10/dist-packages/torch/lib /opt/pytorch/lib \
    && ln -s /usr/local/lib/python3.10/dist-packages/torch/share /opt/pytorch/share \
    && test -s /opt/pytorch/share/cmake/Torch/TorchConfig.cmake \
    && test -s /opt/pytorch/include/torch/csrc/api/include/torch/torch.h \
    && test -s /opt/pytorch/lib/libtorch.so

COPY tests/requirements-vm3.txt /tmp/requirements-vm3.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-vm3.txt

WORKDIR /work
