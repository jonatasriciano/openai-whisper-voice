import datetime
import logging

# Suppress third-party debug logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("pydub.converter").setLevel(logging.ERROR)
import asyncio
from app.core.conversation import main_loop

def log_step(message):
    print(f"🕒 [{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")
if __name__ == "__main__":
    try:
        log_step("🤖 Starting voice assistant...")
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        log_step("🛑 Assistant interrupted by user")
    finally:
        log_step("✅ Assistant terminated cleanly")