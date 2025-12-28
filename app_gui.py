import streamlit as st
import os
import time
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

# Import modul buatanmu sendiri
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

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="BioPsy Voice Assistant", page_icon="otak.png")

# --- SETUP SESSION STATE (Memori Chat) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo, aku BioPsy😆👋🏼 Ada yang ingin kamu ceritakan hari ini? 🤗"}
    ]

# --- SIDEBAR (Menu Samping) ---
with st.sidebar:
    st.image("otak.png", width=100)
    # Trik CSS dikit biar nempel banget (Negative Margin)
    st.markdown("""
        <div style="margin-top: -20px;">
            <h3>Tentang BioPsy</h3>
        </div>
    """, unsafe_allow_html=True) 
    st.info(
        "Asisten virtual CBT (Cognitive Behavioural Therapy) "
        "yang siap mendengarkan cerita dan keluh kesahmu tanpa menghakimi."
    )
    
# --- FITUR 1: MOOD METER (REAL-TIME) ---
    st.markdown("---")
    st.subheader("📊 Kondisi Emosional")
    
    # Logika Simpel: Cek kata-kata di pesan TERAKHIR user
    # (Ini simulasi cerdas tanpa perlu panggil AI mahal-mahal)
    last_mood_score = 5 # Default netral
    
    if st.session_state.messages:
        # Ambil pesan terakhir dari User (bukan Bot)
        last_user_msg = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "")
        last_user_msg = last_user_msg.lower()
        
        # Deteksi kata kunci sederhana
        if any(w in last_user_msg for w in ["sedih", "takut", "cemas", "bingung", "sakit", "capek", "lelah", "mati"]):
            last_mood_score = 8 # Zona Merah
        elif any(w in last_user_msg for w in ["marah", "kesal", "benci", "sebal"]):
            last_mood_score = 7 # Zona Oranye
        elif any(w in last_user_msg for w in ["senang", "bahagia", "tenang", "lega", "makasih", "baik"]):
            last_mood_score = 2 # Zona Hijau
        else:
            last_mood_score = 5 # Netral
            
    # Tampilkan Slider (Otomatis berubah sesuai score di atas)
    mood_display = st.progress(last_mood_score / 10)
    
    if last_mood_score >= 8:
        st.caption("Status: **Perlu Perhatian** 🔴")
    elif last_mood_score <= 3:
        st.caption("Status: **Stabil / Positif** 🟢")
    else:
        st.caption("Status: **Netral / Sedang** 🟠")


    # --- FITUR 2: DOWNLOAD REKAM MEDIS (REAL-TIME) ---
    st.markdown("---")
    st.subheader("📥 Dokumentasi")

    # Susun teks chat log dari awal sampai akhir
    chat_log_text = "RIWAYAT SESI KONSELING 13CBT\n"
    chat_log_text += "============================\n\n"
    
    for msg in st.session_state.messages:
        role = "PASIEN" if msg["role"] == "user" else "TERAPIS (AI)"
        chat_log_text += f"[{role}]: {msg['content']}\n\n"

    # Tombol Download (Selalu update isinya)
    st.download_button(
        label="Simpan Chat Log (.txt)",
        data=chat_log_text,
        file_name="Rekam_Medis_13CBT.txt",
        mime="text/plain",
        help="Klik untuk menyimpan seluruh percakapan sejauh ini."
    )

    st.markdown("---") # Garis pembatas
    
    st.warning(
        "⚠️ **PENTING:**\n"
        "Aplikasi ini bukan pengganti psikolog klinis. "
        "Jika kamu dalam bahaya atau krisis, segera hubungi layanan darurat! \n\n📞Layanan Darurat 119\n\n📞Halo Kemenkes: 1500-567"
    )
       
    st.markdown("---")
    
    st.caption("© 2025 Kelompok 13 - Kecerdasan Buatan Biomedik")
    
    # List Developer dengan Link LinkedIn
    st.caption("""[Rosi](https://www.linkedin.com/in/rosianaf-puspita/) [Aida](https://www.linkedin.com/in/rufaidakariemah/) [Caca](https://www.linkedin.com/in/grace-kezia-siregar-8781ab36b/)
        """
    )

# Bikin 2 kolom: Kecil untuk gambar, Besar untuk judul
col1, col2 = st.columns([0.4, 5]) 

with col1:
    # Tampilkan gambar lokal (atur width biar pas)
    st.image("otak.png", width=80) 

with col2:
    # Judulnya (Hapus emoji otaknya karena sudah diganti gambar)
    st.title("BioPsy: Teman Cerita Kamu")

st.markdown("*Voice-based Cognitive Behavioural Therapy Assistant*")

# --- INISIALISASI SYSTEM (Cuma jalan sekali) ---
@st.cache_resource
def setup_system():
    """Load config dan retriever sekali saja biar gak berat."""
    cfg = Config()
    os.makedirs(cfg.TMP_DIR, exist_ok=True)
    
    # Cek index dataset
    force_rebuild = os.getenv("FORCE_REBUILD", "0") == "1"
    ensure_index(cfg, force_rebuild=force_rebuild)
    
    # Load retriever
    retriever = CBTRetriever(cfg)
    return cfg, retriever

cfg, retriever = setup_system()

# --- FUNGSI UTAMA ---
def process_voice_input():
    """Handle proses rekam -> STT -> RAG -> LLM -> TTS"""
    
    # Path file sementara
    in_wav = os.path.join(cfg.TMP_DIR, "user_gui.wav")
    out_audio = os.path.join(cfg.TMP_DIR, f"assistant_gui_{int(time.time())}.mp3") # Pake timestamp biar browser gak cache file lama

    # 1. REKAM SUARA
    with st.spinner("🎙️ Mendengarkan... (Bicara sekarang)"):
        # Kita pakai durasi dari config, atau bisa di-hardcode misal 5 detik
        record_wav(in_wav, seconds=cfg.RECORD_SECONDS, sample_rate=cfg.SAMPLE_RATE)
    
    st.success("✅ Selesai merekam. Memproses...")

    # 2. SPEECH TO TEXT
    user_text = transcribe_audio(in_wav, model=cfg.STT_MODEL)
    user_text = (user_text or "").strip()

    if not user_text:
        st.warning("Suara tidak terdengar jelas. Coba lagi ya.")
        return

    # Tampilkan chat user
    st.session_state.messages.append({"role": "user", "content": user_text})

    # 3. CEK INTENT KHUSUS (Stop / Filler / Safety)
    
    # A) Stop Intent
    if is_stop_intent(user_text):
        reply = "Oke, kita berhenti dulu ya. Terima kasih sudah cerita. Jaga diri baik-baik."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        speak_text(reply, out_audio, model=cfg.TTS_MODEL, voice=cfg.TTS_VOICE)
        st.audio(out_audio, autoplay=True)
        return

    # B) Filler (Gumaman)
    if is_mostly_filler(user_text):
        reply = "Aku denger kok. Pelan-pelan aja ceritanya."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        speak_text(reply, out_audio, model=cfg.TTS_MODEL, voice=cfg.TTS_VOICE)
        st.audio(out_audio, autoplay=True)
        return

    # C) Safety Check
    if cfg.ENABLE_SAFETY and safety_check(user_text):
        reply = safety_reply()
        st.session_state.messages.append({"role": "assistant", "content": reply})
        speak_text(reply, out_audio, model=cfg.TTS_MODEL, voice=cfg.TTS_VOICE)
        st.audio(out_audio, autoplay=True)
        return

    # 4. RAG & LLM (Inti Proses)
    with st.spinner("🧠 Sedang berpikir..."):
        # Retrieve context
        examples = retriever.search(user_text, k=cfg.TOP_K)
        
        # Generate jawaban
        messages_payload = build_messages(user_text, examples)
        reply = chat_completion(messages_payload, model=cfg.CHAT_MODEL, temperature=0.4)
    
    # ... (setelah reply didapat) ...

    # Fitur untuk Lihat referensi yang dipakai
    with st.expander("🔍 Debug: Lihat Referensi"):
        st.json(examples) # Menampilkan raw data referensi yang ditemukan

    # 5. TEXT TO SPEECH & TAMPILKAN
    # Simpan jawaban DAN data referensi (examples) ke memori
    st.session_state.messages.append({
        "role": "assistant", 
        "content": reply, 
        "debug_info": examples  # <--- Ini kuncinya!
    })
    
    # Generate suara
    speak_text(reply, out_audio, model=cfg.TTS_MODEL, voice=cfg.TTS_VOICE)
    
    # Putar suara otomatis
    st.audio(out_audio, format="audio/mp3", autoplay=True)


# --- TAMPILAN CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Cek apakah pesan ini punya data rahasia (debug_info)
        if "debug_info" in msg:
            with st.expander("🔍 Debug: Lihat Referensi"):
                st.json(msg["debug_info"])

# --- TOMBOL INPUT ---
# Kita taruh tombol di bawah chat
st.divider()
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🎙️ Mulai Bicara", type="primary", use_container_width=True):
        process_voice_input()
        st.rerun() # Refresh halaman untuk menampilkan chat baru

with col2:
    st.caption(f"Klik tombol dan bicara selama {cfg.RECORD_SECONDS} detik.")