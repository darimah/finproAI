# src/audio/record.py
import numpy as np
import sounddevice as sd
import soundfile as sf
from collections import deque


def record_wav_vad(
    path: str,
    sample_rate: int = 16000,
    max_seconds: int = 60,

    # Stop rules
    silence_seconds: float = 2.0,          # toleransi hening setelah user mulai bicara
    min_record_seconds: float = 4.0,       # jangan stop sebelum minimal durasi ini (lebih aman untuk "mmm... lanjut")

    # Detection rules
    rms_threshold: float = 0.006,          # fallback threshold (dipakai kalau adaptif mati)
    use_adaptive_threshold: bool = True,
    noise_calibration_seconds: float = 0.6,  # ambil noise floor di awal

    # Robustness
    pre_roll_seconds: float = 0.35,        # simpan audio sebelum speech start
    hangover_seconds: float = 0.35,        # setelah RMS turun, kasih "hangover" dulu sebelum dihitung hening beneran
    chunk_ms: int = 30,                    # chunk lebih kecil -> lebih responsif (30ms)
):
    """
    Record until:
    - user started speaking, AND
    - then silence lasts >= silence_seconds, AND
    - total recorded duration >= min_record_seconds
    OR max_seconds reached.

    Improvements:
    - adaptive threshold (noise floor)
    - hangover (pause pendek tidak bikin cepat stop)
    - pre-roll (awal kata tidak kepotong)
    """

    print("🎙️ Recording... (bicara sekarang, akan berhenti otomatis saat hening)")

    chunk_size = int(sample_rate * (chunk_ms / 1000.0))
    max_chunks = int((max_seconds * 1000) / chunk_ms)

    silence_chunks_needed = int((silence_seconds * 1000) / chunk_ms)
    min_chunks_needed = int((min_record_seconds * 1000) / chunk_ms)
    pre_roll_chunks = max(1, int((pre_roll_seconds * 1000) / chunk_ms))
    hangover_chunks = max(1, int((hangover_seconds * 1000) / chunk_ms))

    frames = []
    pre_roll = deque(maxlen=pre_roll_chunks)

    started = False
    silent_chunks = 0
    hangover_left = 0

    # ---------- (1) Noise calibration (buat adaptive threshold) ----------
    noise_rms_values = []
    calib_chunks = int((noise_calibration_seconds * 1000) / chunk_ms)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        for _ in range(calib_chunks):
            chunk, _ = stream.read(chunk_size)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            noise_rms_values.append(rms)
            pre_roll.append(chunk)

        noise_floor = float(np.median(noise_rms_values)) if noise_rms_values else 0.0

        # adaptive: threshold = noise_floor * factor + margin
        # factor 2.5–4 cocok; margin kecil biar gumaman masih kedeteksi
        if use_adaptive_threshold:
            adaptive_threshold = max(rms_threshold, noise_floor * 3.0 + 0.0015)
        else:
            adaptive_threshold = rms_threshold

        # ---------- (2) Main recording loop ----------
        for i in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            pre_roll.append(chunk)

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            # detect speech start
            if not started:
                if rms >= adaptive_threshold:
                    started = True
                    # include pre-roll so we don't cut initial phonemes
                    frames.extend(list(pre_roll))
                    pre_roll.clear()
                    # reset counters
                    silent_chunks = 0
                    hangover_left = hangover_chunks
                continue

            # after started: store
            frames.append(chunk)

            # hangover logic: ketika rms turun, jangan langsung hitung hening
            if rms >= adaptive_threshold:
                silent_chunks = 0
                hangover_left = hangover_chunks
            else:
                # kalau hangover masih ada, kurangi dulu; belum hitung silence
                if hangover_left > 0:
                    hangover_left -= 1
                else:
                    silent_chunks += 1

            # stop if enough silence and min duration met
            if i >= min_chunks_needed and silent_chunks >= silence_chunks_needed:
                break

    # ---------- (3) Save ----------
    if frames:
        audio = np.concatenate(frames, axis=0)
    else:
        # kalau user gak bicara sama sekali, simpan pre-roll biar file tetap valid
        if len(pre_roll) > 0:
            audio = np.concatenate(list(pre_roll), axis=0)
        else:
            audio = np.zeros((int(sample_rate * 0.5), 1), dtype=np.float32)

    sf.write(path, audio, sample_rate)
    print(f"✅ Saved: {path}")


# Wrapper kompatibel app.py
def record_wav(path: str, seconds: int = 10, sample_rate: int = 16000):
    """
    seconds di app.py kita anggap sebagai MAX seconds.
    Settings default dibuat lebih 'tahan' untuk gumaman + pause.
    """
    return record_wav_vad(
        path=path,
        sample_rate=sample_rate,
        max_seconds=seconds,

        silence_seconds=2.0,        # toleransi pause
        min_record_seconds=3.5,     # jangan stop terlalu cepat

        rms_threshold=0.006,        # fallback
        use_adaptive_threshold=True,
        noise_calibration_seconds=0.6,

        pre_roll_seconds=0.35,
        hangover_seconds=0.35,
        chunk_ms=30,
    )
