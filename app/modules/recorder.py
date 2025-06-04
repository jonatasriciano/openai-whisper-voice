import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import time
import wave
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import datetime
import logging
logging.basicConfig(level=logging.DEBUG)

def log_step(message):
    print(f"🕒 [{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")

log_step("🎬 Recorder initialized")

from app.core.config import SAMPLE_RATE, BLOCKSIZE, SILENCE_TIMEOUT, OUTPUT_FILE, CHANNELS, MIC_DEVICE

def auto_select_input_device():
    devices = sd.query_devices()
    input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    if not input_devices:
        print("❌ No input devices found.")
        exit(1)

    print("🎤 Available input devices:")
    for idx, dev in input_devices:
        print(f"{idx}: {dev['name']}")

    while True:
        try:
            selection = int(input("🔧 Select input device index: "))
            if any(idx == selection for idx, _ in input_devices):
                return selection
            else:
                print("❌ Invalid selection. Try again.")
        except ValueError:
            print("❌ Please enter a valid number.")


def record_audio(channels=None, mic_device=None):
    """
    Record audio using the microphone and save it to OUTPUT_FILE.

    Args:
        channels (int, optional): Number of channels to record. Defaults to value from settings.
        mic_device (int, optional): Device index to use. Defaults to value from settings.

    Returns:
        bool: True if recording successful and non-silent, False otherwise.
    """
    log_step("🧪 Entering record_audio function")
    log_step("Starting audio recording")
    if channels is None:
        channels = CHANNELS
    if mic_device is None:
        mic_device = MIC_DEVICE
    print("🎙️ Recording... Speak naturally.")
    q = queue.Queue()
    silence_threshold = 0.007
    silence_duration = SILENCE_TIMEOUT
    block_duration = 0.1
    block_size = int(SAMPLE_RATE * block_duration)
    max_blocks = int(SAMPLE_RATE * 30 / block_size)
    silence_blocks_required = int(silence_duration / block_duration)
    silence_start_time = None
    frames = []

    def callback(indata, frames_count, time_info, status):
        nonlocal silence_start_time
        if indata is None or len(indata) == 0:
            log_step("⚠️ Empty audio block received")
            return
        q.put(indata.copy())
        normalized = indata.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(np.square(normalized)))
        log_step(f"Block RMS: {rms:.6f}")
        if np.isnan(rms):
            rms = 0.0
        if rms < silence_threshold:
            if silence_start_time is None:
                silence_start_time = time.time()
                log_step("🔇 Detected silence start")
            else:
                log_step(f"🔇 Continuing silence... elapsed: {time.time() - silence_start_time:.2f}s")
        else:
            if silence_start_time is not None:
                log_step(f"🎙️ Voice resumed after {time.time() - silence_start_time:.2f}s of silence")
            silence_start_time = None

    try:
        log_step("Opening sounddevice InputStream")
        with sd.InputStream(callback=callback, channels=channels, samplerate=SAMPLE_RATE, dtype='int16', device=mic_device, blocksize=BLOCKSIZE):
            for i in range(max_blocks):
                block = q.get()
                frames.append(block)
                log_step(f"Captured audio block {i+1}")
                if silence_start_time is not None and (time.time() - silence_start_time > SILENCE_TIMEOUT):
                    elapsed_silence = time.time() - silence_start_time
                    log_step(f"Elapsed silence: {elapsed_silence:.2f}s (threshold: {SILENCE_TIMEOUT}s)")
                    print("🔇 Silence detected. Stopping.")
                    log_step("Silence timeout reached, stopping recording")
                    break
        if not frames:
            log_step("❌ No audio frames captured. Aborting recording.")
            log_step("🏁 Exiting record_audio function")
            return False
        audio_data = np.concatenate(frames)
        sf.write(OUTPUT_FILE, audio_data, SAMPLE_RATE)
        log_step(f"✅ Audio file written to {OUTPUT_FILE}, length: {len(audio_data) / SAMPLE_RATE:.2f}s")
        log_step("Validating if audio is silent")
        if is_audio_file_silent(OUTPUT_FILE) and channels == 1:
            info = sd.query_devices(mic_device, 'input')
            if info['max_input_channels'] >= 2:
                log_step("Retrying in stereo mode")
                print("🔄 Retrying in stereo...")
                log_step("🏁 Exiting record_audio function")
                return record_audio(channels=2, mic_device=mic_device)
            else:
                print("❌ Stereo not supported.")
                log_step("Finished audio recording")
                log_step("🏁 Exiting record_audio function")
                return False
        elif is_audio_file_silent(OUTPUT_FILE) and channels == 2:
            print("❌ Silent audio in stereo. Check your mic.")
            log_step("Finished audio recording")
            log_step("🏁 Exiting record_audio function")
            return False
        else:
            log_step("Finished audio recording")
            log_step("🏁 Exiting record_audio function")
            return True
    except KeyboardInterrupt:
        print("🛑 Interrupted.")
        log_step("Finished audio recording")
        log_step("🏁 Exiting record_audio function")
        return False
    finally:
        log_step("🔧 Cleaning up sounddevice stream")
        try:
            current_stream = sd.get_stream()
            if current_stream and current_stream.active:
                current_stream.stop()
        except Exception as e:
            log_step(f"⚠️ Error stopping stream: {e}")

def is_audio_file_silent(path):
    log_step(f"Checking if audio file {path} is silent")
    with wave.open(path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
        if samples.size == 0:
            log_step("Silent check completed")
            return True
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        log_step(f"📊 Audio RMS calculated: {rms:.2f}")
        log_step("Silent check completed")
        return rms < 10