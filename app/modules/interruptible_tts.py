import datetime
import numpy as np
import os
import simpleaudio as sa
import sounddevice as sd
import threading
import time

from app.config.settings import INTERRUPTION_THRESHOLD, CHUNK_DURATION, INTERRUPTION_TIMEOUT, DEBUG_INTERRUPTION

"""
Handles audio playback and allows user interruption by speaking over it.
Uses microphone RMS monitoring to detect interruptions and stop playback.
"""

class AudioInterrupter:
    def __init__(self, audio_path: str, device: int = None):
        self.audio_path = audio_path
        self.device = device
        self.stop_event = threading.Event()
        self._thread = threading.Thread(target=self._play)
        self.interrupted = False

    def _play(self):
        log_step("Starting audio playback")
        try:
            log_step("Loading WAV audio file")
            wave_obj = sa.WaveObject.from_wave_file(self.audio_path)
            if not self.audio_path.lower().endswith(".wav"):
                log_step("❌ Unsupported format: interruptible_tts only supports WAV files.")
                return
            play_obj = wave_obj.play()
            log_step("Audio playback started")
            while not self.stop_event.is_set() and play_obj.is_playing():
                time.sleep(0.1)
            play_obj.stop()
            log_step("Audio playback stopped")
        except Exception as e:
            log_step(f"❌ Audio playback error: {e}")

    def _monitor_mic(self):
        # Callback that monitors microphone input and checks for loudness above threshold
        log_step("Starting microphone monitoring")
        def callback(indata, frames, time_, status):
            if status:
                log_step(f"⚠️ Mic status: {status}")
            rms = np.sqrt(np.mean(indata**2)) * 1000
            if DEBUG_INTERRUPTION:
                log_step(f"🔊 RMS: {rms:.2f}")
            if rms > INTERRUPTION_THRESHOLD:
                self.interrupted = True
                log_step("Voice interruption detected")
                self.stop_event.set()

        # Start microphone stream and listen for interruption signals
        with sd.InputStream(callback=callback, channels=1, samplerate=16000, blocksize=int(16000 * CHUNK_DURATION), device=self.device):
            timeout = time.time() + INTERRUPTION_TIMEOUT
            while not self.stop_event.is_set() and time.time() < timeout:
                time.sleep(CHUNK_DURATION)

    def start(self):
        log_step("Starting playback and mic monitor threads")
        mic_thread = threading.Thread(target=self._monitor_mic)
        self._thread.start()
        mic_thread.start()
        self._thread.join()
        mic_thread.join()
        log_step("Playback and mic monitoring completed")

def play_audio_interruptible(audio_path: str, device: int = None) -> bool:
    log_step(f"Invoking interruptible playback for {audio_path}")
    player = AudioInterrupter(audio_path, device)
    player.start()
    log_step(f"Playback interrupted: {player.interrupted}")
    return player.interrupted
