# hash:sha256:5d57d97bc8b457cb1d9ec1a01bd8450bbbaee54e84cb166591966ad9a75146d2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/ubuntu-desktop-pytorch:2.1.0-cuda11.8-mambaforge23.1.0-4-python3.10.12-fiji20231107-1617-firefox120.0-igv2.16.1-guacamole1.5.2-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    cellxgene-census==1.16.1 \
    chromadb==0.5.7 \
    copasi-basico==0.75 \
    langchain==0.3.0 \
    langchain-community==0.3.0 \
    langchain-ollama==0.2.0 \
    langchain-openai==0.2.2 \
    ollama==0.3.3 \
    pandas==2.2.3 \
    pyarrow==17.0.0 \
    python-libsbml==5.20.4 \
    requests==2.32.3 \
    streamlit==1.39.0 \
    tabulate==0.9.0 \
    tqdm==4.66.5

COPY postInstall /
RUN /postInstall
