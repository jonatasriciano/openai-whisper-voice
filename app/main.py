import datetime
import asyncio
from app.core.conversation import start_conversation

def log_step(message):
    print(f"🕒 [{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")
if __name__ == "__main__":
    try:
        log_step("Launching start_conversation with asyncio")
        asyncio.run(start_conversation())
    except KeyboardInterrupt:
        log_step("Conversation interrupted by user")
    finally:
        log_step("Program terminated")