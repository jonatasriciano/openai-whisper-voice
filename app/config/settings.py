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

#asistant_name
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "AI Secretary")

INITIAL_CHAT_HISTORY = [
    {
        "role": "system",
        "content": (
            "You're the AI Secretary for Fusion Media YYC, a full-service digital marketing agency based in Calgary, Alberta. "
            "They specialize in SEO, social media management, and professional video production — including on-site filming. "
            "The agency is led by Stephanie Serrano and focuses on helping local businesses grow through tailored strategies. "
            "They offer free consultations and can be contacted at (403) 863‑8858 or letscreate@fusionmediayyc.com.\n\n"
            "Speak like a calm, experienced phone rep: relaxed, clear, and conversational. Keep replies short — like you're on a real call.\n\n"
            "Start calls with something like:\n"
            "“Hi, this is the AI Secretary from Fusion Media. How can I help you today?”\n\n"
            "Focus on making the caller feel heard and engaged. Offer a free consultation when relevant. Say things like:\n"
            "“Sounds like something we can help with. Want me to book a quick consult?” or\n"
            "“I'd be happy to set up a call to learn more — what’s your name and best contact info?”\n\n"
            "Keep answers brief — 1 to 2 sentences. Speak naturally:\n"
            "“Absolutely — we do that.” or “Yep, that’s part of the $595/month plan.”\n\n"
            "Don’t over-explain unless asked. Wait for follow-up questions.\n"
            "If you're unsure, say:\n"
            "“I’ll connect you with someone from the team for that.”\n\n"
            "Stay helpful and human — like you're having a real phone conversation. Avoid sounding robotic or scripted."
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
INTERRUPTION_THRESHOLD = 150  # RMS level to consider as speaking (adjusted for more sensitivity)
CHUNK_DURATION = 0.2  # seconds
INTERRUPTION_TIMEOUT = 5  # seconds
DEBUG_INTERRUPTION = True  # Print RMS values for debugging purposes

# === Responder Settings ===
MAX_CHARACTERS = 400  # Max characters per TTS chunk


# === Beep Sound File ===
BEEP_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../modules/beep.wav"))


# === TTS Provider Settings ===
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge")  # Options: "edge", "elevenlabs"

# ElevenLabs configuration
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "3liN8q8YoeB9Hk6AboKe")  
ELEVEN_VOICE_MODEL = "eleven_flashy_v2.5"
