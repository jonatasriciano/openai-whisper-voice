import simpleaudio as sa
import os
from config.settings import BEEP_FILE

"""
Provides utility functions for simple audio effects such as playing a beep.
"""

def play_beep():
    try:
        wave_obj = sa.WaveObject.from_wave_file(BEEP_FILE)
        wave_obj.play()
    except Exception as e:
        print(f"⚠️ Failed to play beep sound. Error: {e}")
