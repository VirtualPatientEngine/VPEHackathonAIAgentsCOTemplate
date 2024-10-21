# hash:sha256:4d9cbf4859850b76a2d019f5e54ab8a3a1c806f8499df4ccf393a296290ef59a
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
    streamlit==1.38.0 \
    tabulate==0.9.0 \
    tqdm==4.66.5

COPY postInstall /
RUN /postInstall
