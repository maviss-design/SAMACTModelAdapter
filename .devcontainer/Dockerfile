FROM pytorch/pytorch:2.12.1-cuda13.2-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# RUN python -m pip install --upgrade \
#     pip \
#     setuptools \
#     wheel
