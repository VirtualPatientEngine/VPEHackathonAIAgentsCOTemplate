# hash:sha256:60d0817d11204e6fe0b6f7852da218545f446c84f194835d6a70156b13664df8
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/ubuntu-desktop-pytorch:2.1.0-cuda11.8-mambaforge23.1.0-4-python3.10.12-fiji20231107-1617-firefox120.0-igv2.16.1-guacamole1.5.2-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    cellxgene-census==1.16.1 \
    chromadb==0.5.7 \
    langchain==0.3.0 \
    ollama==0.3.3 \
    pandas==2.2.3 \
    pyarrow==17.0.0 \
    requests==2.32.3 \
    streamlit==1.39.0 \
    tabulate==0.9.0 \
    tqdm==4.66.5

COPY postInstall /
RUN /postInstall
