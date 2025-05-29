"""
Centralized settings and environment variable configuration.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# === OpenAI Configuration ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")

openai = OpenAI(api_key=OPENAI_API_KEY)
ASSISTANT_ID = OPENAI_ASSISTANT_ID

INITIAL_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant from Fusion Media YYC."}
]

# === Audio Configuration ===
AUDIO_DIR = "audio"
OUTPUT_FILE = os.path.join(AUDIO_DIR, "input.wav")
os.makedirs(AUDIO_DIR, exist_ok=True)

CHANNELS = 1
SAMPLE_RATE = 16000
BLOCKSIZE = 2048
SILENCE_TIMEOUT = 2.0
MIC_DEVICE = None  # Use default microphone

# === Transcriber (Whisper) Configuration ===
WHISPER_MODEL_NAME = "tiny.en"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_LANGUAGE = "en"
WHISPER_BEAM_SIZE = 5

# === Interruptible TTS Settings ===
INTERRUPTION_THRESHOLD = 250  # RMS level to consider as speaking
CHUNK_DURATION = 0.2  # seconds
INTERRUPTION_TIMEOUT = 5  # seconds

# === Responder Settings ===
MAX_CHARACTERS = 400  # Max characters per TTS chunk

# === Beep Sound File ===
BEEP_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules/beep.wav"))