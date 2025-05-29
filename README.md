# 🗣️ OpenAI Whisper Voice Assistant

A simple Python voice assistant that listens to your speech, transcribes it using Whisper, sends the transcription to OpenAI GPT-4o for a response, and then speaks the response aloud using Edge TTS.

---

## 📁 Project Structure

```
openai-whisper-voice/
├── app/
│   ├── main.py               # Entry point - starts conversation
│   ├── config/               # Settings and API keys
│   │   └── settings.py
│   ├── core/                 # Orchestration logic
│   │   └── conversation.py
│   ├── modules/              # Feature modules
│   │   ├── recorder.py
│   │   ├── responder.py
│   │   ├── transcriber.py
│   │   ├── interruptible_tts.py
│   │   └── audio_utils.py
│   └── utils/                # Shared utilities
│       └── logger.py
├── audio/                    # Temporary audio files
├── logs/                     # Saved user/bot audio logs
├── docker-compose.yml
├── Dockerfile
├── makefile
├── requirements.txt
├── .env                      # API keys (not tracked)
└── tests/
    └── test_main.py
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- `ffmpeg` installed on the host system
- OpenAI API key

---

### 🔧 Installation (local)

```bash
# Clone this repo or download it
cd openai-whisper-voice

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create the .env file
cp .env.example .env  # or create manually
```

Your `.env` file must include:

```
OPENAI_API_KEY=your_key_here
OPENAI_ASSISTANT_ID=your_assistant_id_here
```

---

### ▶️ Run Locally

```bash
make run
```

---

### 🐳 Run with Docker

```bash
docker compose up --build
```

---

## 💡 Features

- Records microphone input with silence detection
- Transcribes using [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- Sends text to OpenAI's GPT-4o
- Speaks response using Microsoft's Edge TTS
- Works entirely offline for recording/transcription

---

## 📜 License

MIT