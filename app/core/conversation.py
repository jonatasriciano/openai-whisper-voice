import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import os
import shutil
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
from app.modules.recorder import record_audio
from app.modules.transcriber import transcribe_with_whisper
from app.modules.responder import respond_and_speak, respond_with_text, speak_with_edge_tts
from app.core.config import OUTPUT_FILE
from app.modules.interruptible_tts import play_audio_interruptible
from app.modules.audio_utils import play_beep


async def start_conversation():
    print("🤖 Voice Assistant is ready. Speak naturally. Press Ctrl+C to stop.\n")
    await respond_and_speak("Hello! I'm listening. You can start speaking.")
    try:
        while True:
            start = datetime.now()
            audio_success = record_audio()
            print(f"⏱️ Step record_audio took {(datetime.now() - start).total_seconds()} seconds")
            if not audio_success:
                print("⚠️ Audio recording failed.")
                continue
            if not os.path.exists(OUTPUT_FILE):
                print("❌ Audio file not found. Check microphone or save logic.")
                continue
            start = datetime.now()
            user_input = transcribe_with_whisper(OUTPUT_FILE)
            print(f"⏱️ Step transcribe_with_whisper took {(datetime.now() - start).total_seconds()} seconds")

            start = datetime.now()
            assistant_text = respond_with_text(user_input)
            print(f"⏱️ Step respond_with_text took {(datetime.now() - start).total_seconds()} seconds")

            start = datetime.now()
            audio_path = await speak_with_edge_tts(assistant_text)
            print(f"⏱️ Step speak_with_edge_tts took {(datetime.now() - start).total_seconds()} seconds")

            if audio_path:
                start = datetime.now()
                interrupted = play_audio_interruptible(audio_path)
                print(f"⏱️ Step play_audio_interruptible took {(datetime.now() - start).total_seconds()} seconds")
                if interrupted:
                    print("⏹️ Assistant interrupted by user speech.")
                    play_beep()
                    continue  # Resume listening immediately
            else:
                print("❌ Skipping playback due to TTS failure.")
            start = datetime.now()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_log_path = os.path.join(LOG_DIR, f"user_{timestamp}.wav")
            output_log_path = os.path.join(LOG_DIR, f"bot_{timestamp}.mp3")

            # Save copies of audio files
            shutil.copyfile(OUTPUT_FILE, input_log_path)
            shutil.copyfile(audio_path, output_log_path)
            print(f"⏱️ Step saving logs took {(datetime.now() - start).total_seconds()} seconds")
            print("\n--- Next Turn ---\n")
    except KeyboardInterrupt:
        print("\n👋 Conversation terminated by user.")
        generate_html_log()


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
