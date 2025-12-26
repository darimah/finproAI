import sounddevice as sd
import soundfile as sf
from src.llm.client import text_to_speech

def speak_text(text: str, out_path: str, model: str, voice: str):
    # Generate audio
    text_to_speech(text, out_path, model=model, voice=voice)

    # Play audio
    data, samplerate = sf.read(out_path, dtype='float32')
    sd.play(data, samplerate)
    sd.wait()
