# TCLD environment image.
# Everything needed to run the code lives here; data, checkpoints and run
# outputs are mounted as volumes (/data, /runs, /ckpt) and never baked in.
#
# Build (from repo root):   make docker-build
# Shell:                    make docker-shell
# Any command:              make docker-run CMD="make smoke"

ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG UID=1000
ARG GID=1000
ARG USERNAME=tcld

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TCLD_DATA=/data \
    TCLD_RUNS=/runs \
    TCLD_CKPT=/ckpt \
    TCLD_IN_DOCKER=1 \
    PYTHONPATH=/workspace:/workspace/external/D-FINE

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
        git make wget curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

# Non-root user matching the host UID/GID so files in mounted volumes stay yours.
RUN (getent group ${GID} || groupadd -g ${GID} ${USERNAME}) \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} \
    && mkdir -p /workspace /data /runs /ckpt \
    && chown -R ${UID}:${GID} /workspace /data /runs /ckpt

WORKDIR /workspace

# Python deps first (better layer caching).
COPY --chown=${UID}:${GID} requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# Code: repo + pinned D-FINE submodule. In dev the repo is bind-mounted over
# /workspace, so this copy only matters for standalone image use.
COPY --chown=${UID}:${GID} . /workspace
RUN pip install --no-deps -e /workspace

USER ${USERNAME}
CMD ["bash"]
