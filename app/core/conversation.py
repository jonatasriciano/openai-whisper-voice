import os
import shutil
from datetime import datetime

from app.config.settings import OUTPUT_FILE
from app.modules.audio_utils import play_beep
from app.modules.interruptible_tts import play_audio_interruptible
from app.modules.recorder import record_audio
from app.modules.responder import respond_and_speak
from app.modules.transcriber import transcribe_with_whisper

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


async def main_loop():
    print("🧠 Voice assistant is now running in continuous mode. Speak naturally. Press Ctrl+C to exit.\n")
    await respond_and_speak("Hello! I'm listening. You can start speaking.")
    try:
        while True:
            audio_success = record_audio()
            if not audio_success or not os.path.exists(OUTPUT_FILE):
                print("⚠️ Audio recording failed or file not found.")
                continue

            try:
                user_input = transcribe_with_whisper(OUTPUT_FILE)
            except Exception as e:
                print(f"❌ Transcription error: {e}")
                continue

            if not user_input.strip():
                print("⚠️ No speech detected in recording.")
                continue

            print(f"👤 You said: {user_input}")
            try:
                await respond_and_speak(user_input)
            except Exception as e:
                print(f"❌ Response error: {e}")
                continue

            save_conversation_log()
            print("\n--- Next Turn ---\n")

    except KeyboardInterrupt:
        print("\n👋 Conversation terminated by user.")
        generate_html_log()


def save_conversation_log():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_log_path = os.path.join(LOG_DIR, f"user_{timestamp}.wav")
    output_log_path = os.path.join(LOG_DIR, f"bot_{timestamp}.mp3")
    shutil.copyfile(OUTPUT_FILE, input_log_path)
    shutil.copyfile("audio/edge_output.mp3", output_log_path)


def generate_html_log():
    html_log = os.path.join(LOG_DIR, "conversation.html")
    with open(html_log, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Conversation Log</title></head><body>\n")
        f.write("<h1>Conversation Log</h1>\n")
        for filename in sorted(os.listdir(LOG_DIR)):
            if filename.endswith(".wav"):
                label = filename.replace("user_", "").replace(".wav", "")
                f.write(f"<p><b>You ({label}):</b><br>")
                f.write(f"<audio controls src='{filename}'></audio></p>\n")
            elif filename.endswith(".mp3"):
                label = filename.replace("bot_", "").replace(".mp3", "")
                f.write(f"<p><b>Assistant ({label}):</b><br>")
                f.write(f"<audio controls src='{filename}'></audio></p>\n")
        f.write("</body></html>\n")
    print(f"📄 HTML log created: {html_log}")
