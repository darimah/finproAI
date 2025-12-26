# src/llm/prompt.py
import re

# =========================
# Safety (minimal)
# =========================
HIGH_RISK_KEYWORDS = [
    "bunuh diri", "suicide", "kill myself", "end my life",
    "self harm", "self-harm", "melukai diri", "mengakhiri hidup",
    "pengen mati", "ingin mati"
]

def safety_check(user_text: str) -> bool:
    t = (user_text or "").lower()
    return any(k in t for k in HIGH_RISK_KEYWORDS)

def safety_reply() -> str:
    return (
        "Terima kasih sudah berani cerita. Aku ikut prihatin kamu sedang ngerasain ini. "
        "Karena ini menyangkut keselamatan, aku nggak bisa menangani ini sendirian. "
        "Tolong hubungi orang terdekat yang kamu percaya sekarang (keluarga/teman), "
        "atau layanan darurat setempat / tenaga profesional kesehatan mental. "
        "Kalau kamu berada dalam bahaya segera, cari bantuan darurat secepatnya. "
        "Kalau kamu mau, kamu bisa bilang: kamu sekarang sendirian atau ada orang di dekatmu?"
    )


# =========================
# Stop intent (untuk "sudah", "stop", dll)
# =========================
STOP_PHRASES_EXACT = {
    "sudah", "udah", "cukup", "stop", "berhenti", "selesai", "done",
    "makas", "terima kasih", "thanks",
    "itu aja", "segitu aja", "sampai sini"
}
STOP_PHRASES_CONTAINS = [
    "udah dulu", "sudah dulu", "cukup dulu", "stop dulu", "selesai dulu",
    "kita stop", "kita selesai", "aku selesai", "aku udahan"
]

def is_stop_intent(text: str) -> bool:
    """
    True jika user mengindikasikan ingin berhenti.
    Dibuat konservatif supaya kata 'sudah' dalam kalimat biasa tidak salah deteksi.
    """
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s]", "", t)  # hapus tanda baca

    if not t:
        return False

    # exact short utterance
    tokens = [x for x in t.split() if x]
    if 1 <= len(tokens) <= 3 and t in STOP_PHRASES_EXACT:
        return True

    # contains clear stop phrases
    if any(p in t for p in STOP_PHRASES_CONTAINS):
        return True

    return False


# =========================
# Filler / gumaman detection
# =========================
FILLER_WORDS = {
    "mm", "mmm", "mmmm", "hmm", "hmmm", "eh", "euh", "uh", "uhh",
    "hah", "oh", "aha", "anu", "eee", "em", "emm", "ya", "yah",
    "iya", "oke", "ok"
}

def is_mostly_filler(text: str) -> bool:
    """
    True jika transkrip kebanyakan filler/gumaman dan belum bermakna.
    Gunakan ini setelah STT sebelum RAG/LLM.
    """
    t = (text or "").lower().strip()
    if not t:
        return True

    t2 = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    tokens = [x for x in t2.split() if x]

    if len(tokens) == 0:
        return True

    # hanya 1-2 token dan semuanya filler
    if len(tokens) <= 2 and all(tok in FILLER_WORDS for tok in tokens):
        return True

    # repetisi pendek seperti "mmmm" / "ehh"
    compact = re.sub(r"\s+", "", t)
    if len(compact) <= 6 and any(ch.isalpha() for ch in compact):
        if any(f in compact for f in ["mm", "mmm", "hmm", "eh", "uh", "hah", "oh", "aha", "em"]):
            return True

    return False


# =========================
# Prompt
# =========================
SYSTEM_PROMPT = """
You are BioPsy, a supportive CBT-oriented therapist with a warm, human tone.
Your goal is to help the user clarify their thoughts, emotions, and behaviors using reflective listening and gentle, open-ended questions.

Rules:
- Respond in Indonesian.
- Do NOT diagnose.
- Do NOT provide medical instructions or emergency procedures.
- Avoid giving direct solutions prematurely.
- Use retrieved examples only to learn STYLE, FLOW, and CBT TECHNIQUE; do NOT copy sentences verbatim.
- When reflecting, reuse the user's exact words or short phrases verbatim when possible.
- Explicitly quote key phrases from the user before asking follow-up questions.
- Do NOT paraphrase important emotional statements unnecessarily.
- Keep responses concise (roughly 5–10 sentences), empathetic, and practical.
- Ask exactly ONE gentle follow-up question at the end.

Conversation repair:
- If the user says you interrupted, responded too quickly, or asks you to wait:
  1) Apologize briefly and sincerely in Indonesian.
  2) Say you will wait and invite them to continue.
  3) Ask only one gentle follow-up question after they share more.
"""


def _format_examples(retrieved_examples: list[dict], max_examples: int = 3) -> str:
    """
    Format contoh RAG agar:
    - konsisten role label: Client/Therapist
    - kompatibel dengan format baru (text/query/response)
    - tetap aman (jangan terlalu panjang)
    """
    if not retrieved_examples:
        return ""

    blocks = []
    for ex in retrieved_examples[:max_examples]:
        # prefer format siap pakai dari ingest
        text = (ex.get("text") or "").strip()

        # fallback jika text tidak ada
        if not text:
            q = (ex.get("query") or ex.get("patient") or "").strip()
            r = (ex.get("response") or ex.get("therapist") or "").strip()
            text = f"Client: {q}\nTherapist: {r}".strip()

        # normalisasi label kalau masih ada "Patient:" dari legacy
        text = text.replace("Patient:", "Client:")

        # metadata ringan (opsional) untuk debugging (LLM diminta tidak menyebut dataset)
        score = ex.get("score")
        if score is not None:
            header = f"Example (similarity={score:.3f}):"
        else:
            header = "Example:"

        blocks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(blocks)


def build_messages(user_text: str, retrieved_examples: list[dict]) -> list[dict]:
    """
    Membuat message list untuk chat_completion.
    Catatan:
    - Stop intent sebaiknya ditangani di app.py (break loop), bukan di sini.
    - Filler detection juga sebaiknya di app.py sebelum masuk LLM.
    """
    user_text = (user_text or "").strip()
    examples_block = _format_examples(retrieved_examples, max_examples=3)

    messages = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]

    if examples_block:
        messages.append({
            "role": "system",
            "content": (
                "Below are examples of real counseling responses. "
                "Learn the STYLE, FLOW, and CBT TECHNIQUE. "
                "Do NOT copy sentences verbatim. "
                "Do NOT mention that these examples exist.\n\n"
                f"{examples_block}"
            )
        })

    messages.append({"role": "user", "content": user_text})
    return messages
