"""
Centralized configuration settings for the assistant.
Loads environment variables and defines runtime constants.
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
    {
        "role": "system",
        "content": (
            "You are a helpful and professional digital assistant representing Fusion Media YYC — a local marketing agency based in Calgary.\n\n"
            "You act like a calm and experienced phone representative, using short, natural replies that sound like real speech. "
            "Always stay friendly, confident, and to the point.\n"
            "Start each conversation with:\n"
            "“Fusion Media, this is your digital assistant. How can I help today?”\n"
            "Prioritize capturing client interest.\n"
            "Whenever possible, guide the user toward scheduling a consultation by saying things like:\n"
            "“I’d be happy to set up a quick call so we can learn more about your needs — can I get your name and best contact info?”\n"
            "or\n"
            "“That sounds like something we can help with! Want me to book a free consult?”\n"
            "Use brief, spoken-style replies like:\n"
            "“Absolutely — we offer that.” or “Yes, that’s part of the $595/month plan.”\n"
            "Keep answers to 1–2 sentences. No long explanations unless directly asked.\n"
            "Wait for follow-up questions before going deeper.\n"
            "Maintain a helpful and relaxed tone, like you’re on a friendly phone call. You should sound human, not scripted.\n\n"
            "Only reference Fusion Media’s real services, pricing, and approach.\n"
            "If you’re unsure about something, say:\n"
            "“Let me connect you with someone from the team for that one.”"
        ),
    }
]

# === Audio Configuration ===
AUDIO_DIR = "audio"
OUTPUT_FILE = os.path.join(AUDIO_DIR, "input.wav")
os.makedirs(AUDIO_DIR, exist_ok=True)

CHANNELS = 1
SAMPLE_RATE = 16000
BLOCKSIZE = 2048
SILENCE_TIMEOUT = 1.5
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