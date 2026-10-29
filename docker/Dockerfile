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

ARG RUNTIME_REQUIREMENTS=docker/requirements.txt
COPY ${RUNTIME_REQUIREMENTS} /tmp/framework-requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/framework-requirements.txt

COPY . /workspace

CMD ["python", "scripts/run.py", "lfm3", "--app", "cli"]
