# hash:sha256:35c9dddb908ddb4758decf4c83ee24f1fe4182f55c9dcaf1ec61dfd5fc360756
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
    faiss-cpu==1.9.0 \
    json-repair==0.30.0 \
    langchain==0.3.0 \
    langchain-community==0.3.0 \
    langchain-experimental==0.3.2 \
    langchain-ollama==0.2.0 \
    langchain-openai==0.2.2 \
    networkx==3.3 \
    ollama==0.3.3 \
    pandas==2.2.3 \
    pubchempy==1.0.4 \
    pyarrow==17.0.0 \
    python-libsbml==5.20.4 \
    rdkit==2024.3.5 \
    requests==2.32.3 \
    scikit-learn==1.5.2 \
    streamlit==1.39.0 \
    tabulate==0.9.0 \
    tqdm==4.66.5

COPY postInstall /
RUN /postInstall
