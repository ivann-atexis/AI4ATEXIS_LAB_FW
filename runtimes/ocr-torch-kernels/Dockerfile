ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3@sha256:f3b4a33fd60d5e0358287bb44d57e91eda45e8f1681fef1bfd6c969371417130
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface \
    GRADIO_ANALYTICS_ENABLED=False \
    PIP_CONSTRAINT="" \
    AI4ATEXIS_LAB_FW_BASE_IMAGE=${BASE_IMAGE}

WORKDIR /workspace
RUN python -m venv --system-site-packages /opt/ai4atexis_lab_fw
ENV PATH="/opt/ai4atexis_lab_fw/bin:${PATH}"

ARG RUNTIME_REQUIREMENTS=runtimes/ocr-torch-kernels/requirements.txt
ARG CAUSAL_CONV1D_REQUIREMENTS=runtimes/ocr-torch-kernels/causal-conv1d-requirements.txt
COPY ${RUNTIME_REQUIREMENTS} /tmp/framework-requirements.txt
COPY ${CAUSAL_CONV1D_REQUIREMENTS} /tmp/causal-conv1d-requirements.txt
# Build against the pinned NGC Torch/CUDA rather than pulling a PyPI Torch into isolation.
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/framework-requirements.txt \
    && MAX_JOBS=4 python -m pip install --no-cache-dir --no-build-isolation --no-deps --require-hashes -r /tmp/causal-conv1d-requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

CMD ["python", "scripts/run.py", "xiaomi-ocr-0", "--app", "cli"]
