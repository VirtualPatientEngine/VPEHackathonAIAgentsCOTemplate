# hash:sha256:99bb1c2738c9121168480f67008528e288aa5647cab93b1cd96b7eb2e0cbbbe4
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/ubuntu-desktop-pytorch:2.1.0-cuda11.8-mambaforge23.1.0-4-python3.10.12-fiji20231107-1617-firefox120.0-igv2.16.1-guacamole1.5.2-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    chromadb==0.5.7 \
    langchain==0.3.0 \
    ollama==0.3.3 \
    pandas==2.2.3 \
    pyarrow==17.0.0 \
    requests==2.32.3 \
    streamlit==1.38.0 \
    tqdm==4.66.5

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server

COPY postInstall /
RUN /postInstall
