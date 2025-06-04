import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from app.core.config import WHISPER_MODEL_NAME, WHISPER_COMPUTE_TYPE, WHISPER_LANGUAGE, WHISPER_BEAM_SIZE
import time
import wave
import numpy as np
from faster_whisper import WhisperModel
from app.utils.debug import log_step

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
            log_step("⚠️ No audio samples found in input file.")
            return "[unrecognized audio]"
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        if np.isnan(rms):
            rms = 0.0
        log_step(f"🔍 Input audio RMS level: {rms:.2f}")
        if rms < 50:
            log_step("⚠️ Audio volume is too low. Check microphone or environment.")

    log_step("📡 Transcribing audio using local Whisper model...")
    start_time = time.time()
    try:
        segments, info = model.transcribe(
            filepath,
            beam_size=WHISPER_BEAM_SIZE,
            language=WHISPER_LANGUAGE,
            vad_filter=False,
            vad_parameters={"threshold": 0.2}
        )
    except Exception as e:
        log_step(f"❌ Transcription error: {e}")
        return "[transcription failed]"

    transcription = "".join([segment.text for segment in segments])
    elapsed = time.time() - start_time

    if not transcription.strip():
        log_step("⚠️ Nothing was transcribed. Audio may be empty or unintelligible.")
        return "[unrecognized audio]"

    log_step(f"⏱️ Transcription took {elapsed:.2f} seconds")
    log_step("📝 You said: " + transcription.strip())

    return transcription.strip()
