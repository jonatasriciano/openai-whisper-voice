import os
import shutil
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
from modules.recorder import record_audio
from modules.transcriber import transcribe_with_whisper
from modules.responder import respond_and_speak, respond_with_text, speak_with_edge_tts
from config.settings import OUTPUT_FILE
from modules.interruptible_tts import play_audio_interruptible
from modules.audio_utils import play_beep


async def start_conversation():
    print("🤖 Voice Assistant is ready. Speak naturally. Press Ctrl+C to stop.\n")
    await respond_and_speak("Hello! I'm listening. You can start speaking.")
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

            assistant_text = respond_with_text(user_input)
            audio_path = await speak_with_edge_tts(assistant_text)
            if audio_path:
                interrupted = play_audio_interruptible(audio_path)
                if interrupted:
                    print("⏹️ Assistant interrupted by user speech.")
                    play_beep()
                    continue  # Resume listening immediately
            else:
                print("❌ Skipping playback due to TTS failure.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_log_path = os.path.join(LOG_DIR, f"user_{timestamp}.wav")
            output_log_path = os.path.join(LOG_DIR, f"bot_{timestamp}.mp3")

            # Save copies of audio files
            shutil.copyfile(OUTPUT_FILE, input_log_path)
            shutil.copyfile(audio_path, output_log_path)
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
