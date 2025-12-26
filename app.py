# app.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

from config import Config

from src.audio.record import record_wav
from src.audio.tts import speak_text
from src.data.dataset_ingest import ensure_index
from src.data.retriever import CBTRetriever
from src.llm.client import transcribe_audio, chat_completion
from src.llm.prompt import (
    build_messages,
    safety_check,
    safety_reply,
    is_mostly_filler,
    is_stop_intent,
)

# (opsional) pesan untuk filler
FILLER_REPLY = (
    "Aku denger kok. Nggak apa-apa kalau kamu lagi mikir atau jeda sebentar. "
    "Lanjutkan aja pelan-pelan, aku dengerin."
)

# (opsional) pesan penutup saat user bilang "sudah/stop"
STOP_REPLY = (
    "Oke, kita berhenti dulu ya. Terima kasih sudah cerita—jaga diri baik-baik. "
    "Kalau kapan-kapan kamu mau lanjut, aku siap dengerin. Sampai ketemu lagi."
)


def main():
    cfg = Config()
    os.makedirs(cfg.TMP_DIR, exist_ok=True)

    # Rebuild index kalau dataset baru ditambahkan:
    # set di .env: FORCE_REBUILD=1
    force_rebuild = os.getenv("FORCE_REBUILD", "0") == "1"

    ensure_index(cfg, force_rebuild=force_rebuild)
    retriever = CBTRetriever(cfg)

    in_wav = os.path.join(cfg.TMP_DIR, "user.wav")
    out_audio = os.path.join(cfg.TMP_DIR, "assistant.mp3")

    print("Voice CBT Chatbot (HOPE+HQC RAG). Ctrl+C untuk keluar.\n")

    while True:
        # A) Record
        record_wav(in_wav, seconds=cfg.RECORD_SECONDS, sample_rate=cfg.SAMPLE_RATE)

        # B) STT
        user_text = transcribe_audio(in_wav, model=cfg.STT_MODEL)
        user_text = (user_text or "").strip()

        if not user_text:
            print("📝 (kosong) Coba ngomong lagi.\n")
            continue

        print(f"📝 You: {user_text}")

        # ✅ B0) STOP INTENT: user bilang "sudah/stop/selesai" -> tutup sesi tanpa tanya lagi
        if is_stop_intent(user_text):
            reply = STOP_REPLY
            print(f"😊 Therapist: {reply}\n")
            speak_text(reply, out_audio, model=cfg.TTS_MODEL, voice=cfg.TTS_VOICE)
            break  # <- keluar dari sesi

        # ✅ B1) FILLER/GUMAMAN: "mmm/eh/hah/oh" -> jangan proses RAG/LLM
        if is_mostly_filler(user_text):
            reply = FILLER_REPLY
            print(f"😊 Therapist: {reply}\n")
            speak_text(reply, out_audio, model=cfg.TTS_MODEL, voice=cfg.TTS_VOICE)
            continue

        # C) Safety gate
        if cfg.ENABLE_SAFETY and safety_check(user_text):
            reply = safety_reply()
            print(f"😊 Therapist: {reply}\n")
            speak_text(reply, out_audio, model=cfg.TTS_MODEL, voice=cfg.TTS_VOICE)
            continue

        # D) Retrieve (gabungan HOPE + HQC)
        examples = retriever.search(user_text, k=cfg.TOP_K)

        # E) LLM
        messages = build_messages(user_text, examples)
        reply = chat_completion(messages, model=cfg.CHAT_MODEL, temperature=0.4)

        print(f"😊 Therapist: {reply}\n")

        # F) TTS
        speak_text(reply, out_audio, model=cfg.TTS_MODEL, voice=cfg.TTS_VOICE)


if __name__ == "__main__":
    main()
