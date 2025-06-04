import os
import time
import wave
import numpy as np
import sounddevice as sd
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from dotenv import load_dotenv
from pydub import AudioSegment
from faster_whisper import WhisperModel
import queue
import soundfile as sf
import asyncio
import edge_tts

BLOCKSIZE = 2048  # Larger audio block to avoid chopping and missing words
SILENCE_TIMEOUT = 1.7  # Seconds of real silence before stopping recording

def auto_select_input_device():
    import sounddevice as sd
    import numpy as np
    print("\n🔎 Testing default input device...")
    try:
        fs = 16000
        duration = 0.8
        print("   Recording a short test...")
        test_audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16', device=None)
        sd.wait()
        rms = np.sqrt(np.mean(test_audio.astype(np.float32) ** 2))
        print(f"   Default device RMS: {rms:.2f}")
        if rms > 50:
            print("✅ Default input device works.")
            return None
        else:
            print("⚠️ Default input device detected very low volume (RMS < 50). Will prompt for selection.")
    except Exception as e:
        print(f"⚠️ Error using default input device: {e}")
        print("   Will prompt for selection.")
    devices = sd.query_devices()
    input_devices = [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    print("\nAvailable audio input devices:")
    for i, name in input_devices:
        print(f"  [{i}] {name}")
    if not input_devices:
        print("❌ No input devices found. Check your microphone.")
        exit(1)
    elif len(input_devices) == 1:
        print(f"Using only available device: {input_devices[0][1]} (index {input_devices[0][0]})")
        return input_devices[0][0]
    else:
        idx = input("Select the input device index for your microphone (default 0): ")
        try:
            idx = int(idx)
        except Exception:
            idx = input_devices[0][0]
        return idx

MIC_DEVICE = None
MIC_DEVICE = auto_select_input_device()

# Query device info early so it's accessible in record_audio
device_info = sd.query_devices(MIC_DEVICE, 'input')

# Maintain conversation history for context
chat_history = [
    {
        "role": "system",
        "content": (
            "You are a helpful and professional digital assistant representing Fusion Media YYC — "
            "a local marketing agency based in Calgary.\n\n"
            "You act like a calm and experienced phone representative, using short, natural replies that sound like real speech. "
            "Always stay friendly, confident, and to the point.\n"
            "• Start each conversation with something like:\n"
            "“Fusion Media, this is your digital assistant. How can I help today?”\n"
            "• Prioritize capturing client interest.\n"
            "Whenever possible, guide the user toward scheduling a consultation by saying things like:\n"
            "“I’d be happy to set up a quick call so we can learn more about your needs — can I get your name and best contact info?”\n"
            "or\n"
            "“That sounds like something we can help with! Want me to book a free consult?”\n"
            "• Use brief, spoken-style replies like:\n"
            "“Absolutely — we offer that.” or “Yes, that’s part of the $595/month plan.”\n"
            "• Keep answers to 1–2 sentences. No long explanations unless directly asked.\n"
            "• Pause often to let the user speak.\n"
            "Wait for follow-up questions before going deeper.\n"
            "• Maintain a helpful and relaxed tone, like you’re on a friendly phone call. You should sound human, not scripted.\n\n"
            "Only reference Fusion Media’s real services, pricing, and approach.\n"
            "If you’re unsure about something, say:\n"
            "“Let me connect you with someone from the team for that one.”"
        )
    }
]

# Load .env with your OpenAI key
load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1  # will attempt stereo (2) if recording is silent
DURATION = 30  # Max recording length in seconds
AUDIO_DIR = "audio"
OUTPUT_FILE = os.path.join(AUDIO_DIR, "input.wav")

# Ensure audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

model = WhisperModel("tiny.en", compute_type="int8")

def record_audio():
    global CHANNELS
    print("🎙️ Recording... Speak naturally. It will stop after a pause.")

    q = queue.Queue()
    duration_limit = 30  # seconds
    silence_threshold = 0.007  # adjusted threshold for normalized RMS
    silence_duration = 2.5   # allows natural pauses
    block_duration = 0.1  # seconds
    block_size = int(SAMPLE_RATE * block_duration)
    max_blocks = int(SAMPLE_RATE * duration_limit / block_size)
    silence_blocks_required = int(silence_duration / block_duration)

    silence_start_time = None
    frames = []

    def callback(indata, frames_count, time_info, status):
        nonlocal silence_start_time
        q.put(indata.copy())
        # print(f"[DEBUG] Frame shape: {indata.shape}, dtype: {indata.dtype}, min: {indata.min()}, max: {indata.max()}")

        # Calculate RMS energy to detect silence (normalized)
        normalized = indata.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(np.square(normalized)))
        if np.isnan(rms):
            rms = 0.0
        # print(f"[DEBUG] RMS value: {rms:.5f}")
        if rms < silence_threshold:
            if silence_start_time is None:
                silence_start_time = time.time()
        else:
            silence_start_time = None

    def is_audio_file_silent(path):
        with wave.open(path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16)
            if samples.size == 0:
                return True
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            print(f"[DEBUG] File RMS after save: {rms:.2f} (should be > 50 for real speech)")
            return rms < 10

    try:
        device_name = sd.query_devices(MIC_DEVICE or sd.default.device['input'])['name'] if MIC_DEVICE is not None else sd.query_devices(sd.default.device['input'])['name']
        print(f"🟢 Listening using device: {device_name}")
        # print("[DEBUG] Attempting to open InputStream...")
        with sd.InputStream(callback=callback, channels=CHANNELS, samplerate=SAMPLE_RATE, dtype='int16', device=MIC_DEVICE, blocksize=BLOCKSIZE):
            print("🟢 Listening...")
            for i in range(max_blocks):
                block = q.get()
                frames.append(block)
                if silence_start_time is not None and (time.time() - silence_start_time > SILENCE_TIMEOUT):
                    print(f"🔇 Detected {SILENCE_TIMEOUT}s of silence. Stopping recording.")
                    break

        audio_data = np.concatenate(frames)
        sf.write(OUTPUT_FILE, audio_data, SAMPLE_RATE)
        print(f"✅ Audio written with soundfile to {OUTPUT_FILE}")

        if os.path.exists(OUTPUT_FILE):
            pass
            # print(f"[DEBUG] File exists after save: YES - Size: {os.path.getsize(OUTPUT_FILE)} bytes")
        else:
            pass
            # print("[DEBUG] File exists after save: NO")

        if is_audio_file_silent(OUTPUT_FILE) and CHANNELS == 1:
            print("⚠️ [WARNING] No sound detected in mono mode. Checking if stereo is supported...")
            device_info = sd.query_devices(MIC_DEVICE, 'input')
            if device_info['max_input_channels'] >= 2:
                print("🔄 Retrying in stereo (2 channels)...")
                CHANNELS = 2
                return record_audio()
            else:
                print("❌ Stereo not supported by this device. Aborting retry.")
                return False
        elif is_audio_file_silent(OUTPUT_FILE) and CHANNELS == 2:
            print("❌ [CRITICAL ERROR] Audio is still silent even after stereo recording. Please check microphone or device selection.")
            return False
        else:
            # Reset CHANNELS to 1 for next time
            CHANNELS = 1
            return True
    except KeyboardInterrupt:
        print("🛑 Recording was manually interrupted.")
        return False

def transcribe_with_whisper(filepath):
    import wave
    # Debug: check RMS level of input audio
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
        beam_size=5,
        language="en",
        vad_filter=False,
        vad_parameters={"threshold": 0.2}
    )
    transcription = "".join([segment.text for segment in segments])
    if not transcription.strip():
        print("⚠️ Nothing was transcribed. Audio may be empty or unintelligible.")
        return "[unrecognized audio]"
    elapsed = time.time() - start_time
    print(f"⏱️ Transcription took {elapsed:.2f} seconds")
    print("\n📝 You said:", transcription.strip())
    return transcription.strip()

def respond_and_speak(user_text):
    if not user_text or user_text.strip() == "":
        print("⚠️ Empty user input. Skipping response.")
        return
    start_time_total = time.time()
    time_after_speech = start_time_total
    print("🧠 Generating response...")
    start_time_gpt = time.time()
    # Maintain conversation context by preserving message history
    global chat_history
    chat_history.append({"role": "user", "content": user_text})

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=chat_history,
        temperature=0.4,
    )
    assistant_text = response.choices[0].message.content.strip()
    MAX_RESPONSE_CHARACTERS = 400
    if len(assistant_text) > MAX_RESPONSE_CHARACTERS:
        print("⚠️ Truncating long assistant response...")
        assistant_text = assistant_text[:MAX_RESPONSE_CHARACTERS].rsplit(".", 1)[0] + "."
    chat_history.append({"role": "assistant", "content": assistant_text})

    elapsed_gpt = time.time() - start_time_gpt
    # print(f"⏱️ GPT chat completion took: {elapsed_gpt:.2f} seconds")
    print("\n🤖 Assistant:", assistant_text)

    # --- Use edge-tts for TTS with English neural voice ---
    async def speak_with_edge_tts(text):
        nonlocal time_after_speech
        print("🔊 Generating speech with edge-tts...")
        time_before_tts_start = time.time()
        ai_response_wait = time_before_tts_start - time_after_speech
        print(f"[DEBUG] Total wait time before audio playback: {ai_response_wait:.2f}s")
        start_time_tts = time.time()
        communicate = edge_tts.Communicate(text, voice="en-CA-ClaraNeural")
        output_path = "audio/edge_output.mp3"
        await communicate.save(output_path)
        # print(f"[DEBUG] edge-tts MP3 saved at: {output_path}")

        # Play the audio with PyDub
        audio = AudioSegment.from_file(output_path, format="mp3").normalize().set_frame_rate(24000)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / np.power(2, 15)
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()
        elapsed_tts = time.time() - start_time_tts
        # print(f"⏱️ edge-tts playback took: {elapsed_tts:.2f} seconds")
        return elapsed_tts

    # Call the async TTS
    elapsed_tts = asyncio.run(speak_with_edge_tts(assistant_text))

    elapsed_total = time.time() - start_time_total
    print(f"⚡️ Spoken Response Delay (TTS only): {elapsed_tts:.2f}s")

    print("🎤 Awaiting user input...")

    return

if __name__ == "__main__":
    print("🤖 Voice Assistant is ready. Speak naturally. Press Ctrl+C to stop.\n")
    chat_history.append({"role": "user", "content": "Hello! I'm listening. You can start speaking."})
    respond_and_speak("Hello! I'm listening. You can start speaking.")
    try:
        while True:
            audio_success = record_audio()
            if not audio_success:
                print("⚠️ Audio recording failed.")
                continue
            if not os.path.exists(OUTPUT_FILE):
                print("❌ Audio file not found. Check microphone or save logic.")
                continue
            user_input = transcribe_with_whisper(OUTPUT_FILE)
            respond_and_speak(user_input)
            print("\n--- Next Turn ---\n")
    except KeyboardInterrupt:
        print("\n👋 Conversation terminated by user.")