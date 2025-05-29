import os
import time
import wave
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf

from config.settings import SAMPLE_RATE, BLOCKSIZE, SILENCE_TIMEOUT, OUTPUT_FILE

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

from config.settings import CHANNELS, MIC_DEVICE

def record_audio(channels=None, mic_device=None):
    """
    Record audio using the microphone and save it to OUTPUT_FILE.

    Args:
        channels (int, optional): Number of channels to record. Defaults to value from settings.
        mic_device (int, optional): Device index to use. Defaults to value from settings.

    Returns:
        bool: True if recording successful and non-silent, False otherwise.
    """
    if channels is None:
        channels = CHANNELS
    if mic_device is None:
        mic_device = MIC_DEVICE
    print("🎙️ Recording... Speak naturally.")
    q = queue.Queue()
    silence_threshold = 0.007
    silence_duration = 2.5
    block_duration = 0.1
    block_size = int(SAMPLE_RATE * block_duration)
    max_blocks = int(SAMPLE_RATE * 30 / block_size)
    silence_blocks_required = int(silence_duration / block_duration)
    silence_start_time = None
    frames = []

    def callback(indata, frames_count, time_info, status):
        nonlocal silence_start_time
        q.put(indata.copy())
        normalized = indata.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(np.square(normalized)))
        if np.isnan(rms):
            rms = 0.0
        if rms < silence_threshold:
            if silence_start_time is None:
                silence_start_time = time.time()
        else:
            silence_start_time = None

    try:
        with sd.InputStream(callback=callback, channels=channels, samplerate=SAMPLE_RATE, dtype='int16', device=mic_device, blocksize=BLOCKSIZE):
            for i in range(max_blocks):
                block = q.get()
                frames.append(block)
                if silence_start_time is not None and (time.time() - silence_start_time > SILENCE_TIMEOUT):
                    print("🔇 Silence detected. Stopping.")
                    break
        audio_data = np.concatenate(frames)
        sf.write(OUTPUT_FILE, audio_data, SAMPLE_RATE)
        if is_audio_file_silent(OUTPUT_FILE) and channels == 1:
            info = sd.query_devices(mic_device, 'input')
            if info['max_input_channels'] >= 2:
                print("🔄 Retrying in stereo...")
                return record_audio(channels=2, mic_device=mic_device)
            else:
                print("❌ Stereo not supported.")
                return False
        elif is_audio_file_silent(OUTPUT_FILE) and channels == 2:
            print("❌ Silent audio in stereo. Check your mic.")
            return False
        else:
            return True
    except KeyboardInterrupt:
        print("🛑 Interrupted.")
        return False

def is_audio_file_silent(path):
    with wave.open(path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
        if samples.size == 0:
            return True
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        return rms < 10