import os
import numpy as np
import faiss
from openai import OpenAI

_client = None

def _client_instance() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        print("DEBUG OPENAI_API_KEY prefix:", (key[:10] if key else None))
        _client = OpenAI(api_key=key)
    return _client

# ---------- Embeddings ----------
def embed_texts(texts: list[str], model: str) -> np.ndarray:
    """
    Returns normalized vectors for cosine similarity (FAISS IP index).
    """
    client = _client_instance()
    r = client.embeddings.create(model=model, input=texts)
    vecs = np.array([d.embedding for d in r.data], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs

def embed_text(text: str, model: str) -> np.ndarray:
    client = _client_instance()
    r = client.embeddings.create(model=model, input=text)
    vec = np.array(r.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

# ---------- STT ----------
def transcribe_audio(wav_path: str, model: str) -> str:
    client = _client_instance()
    with open(wav_path, "rb") as f:
        r = client.audio.transcriptions.create(
            model=model,
            file=f,
        )
    return (r.text or "").strip()

# ---------- Chat ----------
def chat_completion(messages, model: str, temperature: float = 0.4) -> str:
    client = _client_instance()
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return (r.choices[0].message.content or "").strip()

# ---------- TTS ----------
def text_to_speech(text: str, out_path: str, model: str, voice: str):
    client = _client_instance()
    audio = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
    )
    with open(out_path, "wb") as f:
        f.write(audio.read())
