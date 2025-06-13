import os
import asyncio
import logging
import hashlib

import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs

from app.utils.debug import log_step
from app.config.settings import ELEVEN_API_KEY, ELEVEN_VOICE_ID, INITIAL_CHAT_HISTORY

load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs = ElevenLabs(api_key=ELEVEN_API_KEY)

chat_history = INITIAL_CHAT_HISTORY

async def play_audio_file(file_path: str):
    audio = AudioSegment.from_file(file_path, format="mp3").normalize().set_frame_rate(24000)
    audio.export(file_path.replace(".mp3", ".wav"), format="wav")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / np.power(2, 15)
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))

    try:
        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()
    finally:
        sd.stop()
        await asyncio.sleep(0.1)

async def respond_and_speak_stream(user_text: str):
    log_step("🔍 user_text: '{}'".format(user_text))
    log_step("Starting stream-based respond_and_speak")
    if not user_text.strip():
        print("⚠️ Empty input. Skipping.")
        return

    chat_history.append({"role": "user", "content": user_text})
    response_text = ""
    log_step("Calling OpenAI Chat Completion with stream=True")

    stream = openai.chat.completions.create(
        model=os.getenv("OPENAI_MODEL_ID"),
        messages=chat_history,
        stream=True,
        temperature=0.3
    )
    print("🧪 Debug: Receiving stream chunks...")

    full_response = ""
    print("🤖 Assistant: ", end="", flush=True)

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            token = delta.content
            print(token, end="", flush=True)
            full_response += token

    print()  # Newline after full output
    log_step(f"🧠 Full response: {full_response}")

    chat_history.append({"role": "assistant", "content": full_response})

    await speak_with_elevenlabs(full_response)
log_step(f"🔊 Audio path: ")

async def speak_with_elevenlabs(text: str):
    log_step("🔊 Using ElevenLabs TTS via SDK (stream version)")

    hash_name = hashlib.md5(text.encode("utf-8")).hexdigest()
    mp3_path = f"audio/eleven_stream_{hash_name}.mp3"

    if os.path.exists(mp3_path):
        log_step("📁 Using cached audio file")
        await play_audio_file(mp3_path)
        return

    try:
        stream = elevenlabs.text_to_speech.convert(
            text=text,
            voice_id=ELEVEN_VOICE_ID,
            model_id=os.getenv("ELEVEN_MODEL_ID"),
            output_format="mp3_44100_128"
        )
        audio_bytes = b"".join(stream)

        with open(mp3_path, "wb") as f:
            f.write(audio_bytes)

        log_step("✅ Audio file saved. Starting playback.")
        await play_audio_file(mp3_path)

    except Exception as e:
        log_step(f"❌ Error during ElevenLabs streaming TTS: {e}")