from datetime import datetime

def log_step(message):
    print(f"🕒 [{datetime.now().strftime('%H:%M:%S')}] {message}")