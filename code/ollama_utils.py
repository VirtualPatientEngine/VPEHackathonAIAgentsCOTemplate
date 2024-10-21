import os
import ollama
import time
import subprocess


# -------
def restart_ollama_server() -> None:
    try:
        models_list = ollama.list()["models"]
    except:
        process = subprocess.Popen(
            "ollama serve", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(3)


# -------
def ollama_response(message: str, model="llama3.1", role="user", temperature=0.0) -> str:
    response = ollama.chat(
        model="llama3.1",
        options={"temperature": temperature},
        messages=[{"role": role, "content": message}],
    )
    return response["message"]["content"]


# -------
def ollama_stream(message: str, model="llama3.1", role="user", temperature=0.0):
    stream = ollama.chat(
        model=model,
        options={"temperature": temperature},
        messages=[{"role": role, "content": message}],
        stream=True,
    )

    for chunk in stream:
        yield chunk["message"]["content"]


# -------
def get_embedding(input, model="nomic-embed-text"):
    embeds = ollama.embed(model=model, input=input)
    return embeds.get("embeddings", [])


if __name__ == "__main__":
    restart_ollama_server()
    print(ollama_response("Tell me a joke about python"))
    for chunk in ollama_stream("write a 5 sentences song about lammas"):
        print(chunk, end="", flush=True)
