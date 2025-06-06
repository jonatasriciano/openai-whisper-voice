"""Handles interaction with the OpenAI assistant and speech output using edge-tts."""
import asyncio
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import edge_tts
import aiohttp
import datetime
import simpleaudio as sa
def log_step(message):
    print(f"🕒 [{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")

# Import OpenAI client, assistant ID, and initial chat history from settings
from app.config.settings import openai, ASSISTANT_ID, INITIAL_CHAT_HISTORY, MAX_CHARACTERS

# Suppress asyncio coroutine warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*was never retrieved")

# Initialize conversation history from settings
chat_history = INITIAL_CHAT_HISTORY.copy()

async def respond_and_speak(user_text: str):
    """
    Generates a response using OpenAI and speaks it using edge-tts.

    Args:
        user_text (str): The user input to respond to.
    """
    log_step("Starting respond_and_speak")
    if not user_text.strip():
        print("⚠️ Empty input. Skipping response.")
        return

    # Timer for total spoken response delay
    import time
    start_time_total = time.time()

    log_step("Calling OpenAI API for response")
    print("🧠 Generating response...")
    # Update chat_history exactly as in index.py (append user before request)
    chat_history.append({"role": "user", "content": user_text})

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=chat_history,
        temperature=0.4,
    )
    log_step("OpenAI API response received")
    assistant_text = response.choices[0].message.content.strip()

    # Trim long replies for voice output
    if len(assistant_text) > MAX_CHARACTERS:
        assistant_text = assistant_text[:MAX_CHARACTERS].rsplit(".", 1)[0] + "."

    # Append assistant response to chat_history
    chat_history.append({"role": "assistant", "content": assistant_text})
    print("🤖 Assistant:", assistant_text)

    log_step("Sending text to edge-tts")
    # Speak using edge-tts
    try:
        await speak_with_edge_tts(assistant_text)
    except aiohttp.ClientConnectionError as e:
        print(f"🔌 TTS connection failed: {e}. Skipping speech output.")

    elapsed_total = time.time() - start_time_total
    # Print the spoken response delay after TTS completes
    print(f"⚡️ Spoken Response Delay (TTS only): {elapsed_total:.2f}s")

    # Move the print to the end
    print("🎤 Awaiting user input...")

async def speak_with_edge_tts(text: str):
    """
    Uses edge-tts to convert text to speech and play it, with retry logic.

    Args:
        text (str): The text to convert and play.
    """
    log_step("Generating TTS audio file")
    output_path = "audio/edge_output.mp3"
    wav_output_path = "audio/edge_output.wav"
    communicate = edge_tts.Communicate(text, voice="en-CA-ClaraNeural")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        log_step(f"Attempting TTS generation, try {attempt}")
        try:
            # Suppress internal Future warnings by gathering the coroutine
            results = await asyncio.gather(
                communicate.save(output_path),
                return_exceptions=True
            )
            if isinstance(results[0], Exception):
                raise results[0]
            log_step("TTS audio file generated")
            break
        except aiohttp.ClientConnectionError as e:
            print(f"🔁 Attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                print("❌ All TTS attempts failed. Skipping speech.")
                return None
            await asyncio.sleep(1)

    # Convert MP3 to WAV to avoid RIFF error
    audio = AudioSegment.from_file(output_path, format="mp3").normalize().set_frame_rate(24000)
    audio.export(wav_output_path, format="wav")

    # Load the WAV version for playback
    audio = AudioSegment.from_file(wav_output_path, format="wav")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / np.power(2, 15)
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))

    log_step("Playing back audio with sounddevice")
    try:
        log_step(f"▶️ Playing {len(samples)} samples at {audio.frame_rate}Hz")
        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()
    except Exception as e:
        log_step(f"❌ Playback error with sounddevice: {e}")
        try:
            log_step("▶️ Trying fallback playback with simpleaudio")
            play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels,
                                      bytes_per_sample=audio.sample_width,
                                      sample_rate=audio.frame_rate)
            play_obj.wait_done()
        except Exception as fallback_error:
            log_step(f"❌ Fallback playback also failed: {fallback_error}")
    log_step("Releasing sounddevice resources and sleeping briefly to avoid segfault")
    sd.stop()
    await asyncio.sleep(0.1)
    log_step("Audio playback completed")
    return wav_output_path

def respond_with_text(user_input: str) -> str:
    """
    Generates a text-only response using OpenAI's assistant.

    Args:
        user_input (str): The input message from the user.

    Returns:
        str: Assistant's response text.
    """
    log_step("Generating text-only response")
    if not user_input.strip():
        return "I'm sorry, I didn't catch that. Could you please repeat?"

    chat_history.append({"role": "user", "content": user_input})

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=chat_history,
        temperature=0.4,
    )
    log_step("Text-only response received")

    assistant_text = response.choices[0].message.content.strip()
    chat_history.append({"role": "assistant", "content": assistant_text})
    return assistant_text