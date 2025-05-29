from config.settings import WHISPER_MODEL_NAME, WHISPER_COMPUTE_TYPE, WHISPER_LANGUAGE, WHISPER_BEAM_SIZE
import time
import wave
import numpy as np
from faster_whisper import WhisperModel

"""
Handles transcription of recorded audio using a local Whisper model.
"""

# Initialize the local Whisper model
model = WhisperModel(WHISPER_MODEL_NAME, compute_type=WHISPER_COMPUTE_TYPE)

def transcribe_with_whisper(filepath: str) -> str:
    """
    Transcribes the audio from the given file path using Whisper.

    Args:
        filepath (str): Path to the WAV audio file to transcribe.

    Returns:
        str: Transcribed text or fallback string if transcription fails.
    """
    # Check for presence and quality of audio before transcribing
    with wave.open(filepath, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
        if samples.size == 0:
            print("⚠️ No audio samples found in input file.")
            return "[unrecognized audio]"
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        if np.isnan(rms):
            rms = 0.0
        print(f"🔍 Input audio RMS level: {rms:.2f}")
        if rms < 50:
            print("⚠️ Audio volume is too low. Check microphone or environment.")

    print("📡 Transcribing audio using local Whisper model...")
    start_time = time.time()
    segments, info = model.transcribe(
        filepath,
        beam_size=WHISPER_BEAM_SIZE,
        language=WHISPER_LANGUAGE,
        vad_filter=False,
        vad_parameters={"threshold": 0.2}
    )

    transcription = "".join([segment.text for segment in segments])
    elapsed = time.time() - start_time

    if not transcription.strip():
        print("⚠️ Nothing was transcribed. Audio may be empty or unintelligible.")
        return "[unrecognized audio]"

    print(f"⏱️ Transcription took {elapsed:.2f} seconds")
    print("📝 You said:", transcription.strip())

    return transcription.strip()
