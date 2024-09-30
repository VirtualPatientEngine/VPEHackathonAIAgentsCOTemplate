# hash:sha256:8e7539aa44a4e44b722f3730cf71f4a3baa9ff7075f0099ceeec83af045cf2a6
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/ubuntu-desktop-pytorch:2.1.0-cuda11.8-mambaforge23.1.0-4-python3.10.12-fiji20231107-1617-firefox120.0-igv2.16.1-guacamole1.5.2-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN mamba install -y \
        cuda-toolkit==11.8.0 \
        cudnn==8.0.4 \
    && mamba clean -ya

RUN pip install -U --no-cache-dir \
    cellxgene-census==1.16.1 \
    chromadb==0.5.7 \
    langchain==0.3.0 \
    ollama==0.3.3 \
    pandas==2.2.3 \
    pyarrow==17.0.0 \
    requests==2.32.3 \
    tabulate==0.9.0 \
    tqdm==4.66.5
