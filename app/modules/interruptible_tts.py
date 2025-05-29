import threading
import sounddevice as sd
import numpy as np
import time
import simpleaudio as sa

from config.settings import INTERRUPTION_THRESHOLD, CHUNK_DURATION, INTERRUPTION_TIMEOUT

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
        wave_obj = sa.WaveObject.from_wave_file(self.audio_path)
        play_obj = wave_obj.play()
        while not self.stop_event.is_set() and play_obj.is_playing():
            time.sleep(0.1)
        play_obj.stop()

    def _monitor_mic(self):
        # Callback that monitors microphone input and checks for loudness above threshold
        def callback(indata, frames, time_, status):
            if status:
                print(f"⚠️ Mic status: {status}")
            rms = np.sqrt(np.mean(indata**2)) * 1000
            if rms > INTERRUPTION_THRESHOLD:
                self.interrupted = True
                self.stop_event.set()

        # Start microphone stream and listen for interruption signals
        with sd.InputStream(callback=callback, channels=1, samplerate=16000, blocksize=int(16000 * CHUNK_DURATION), device=self.device):
            timeout = time.time() + INTERRUPTION_TIMEOUT
            while not self.stop_event.is_set() and time.time() < timeout:
                time.sleep(CHUNK_DURATION)

    def start(self):
        mic_thread = threading.Thread(target=self._monitor_mic)
        self._thread.start()
        mic_thread.start()
        self._thread.join()
        mic_thread.join()

def play_audio_interruptible(audio_path: str, device: int = None) -> bool:
    player = AudioInterrupter(audio_path, device)
    player.start()
    return player.interrupted
