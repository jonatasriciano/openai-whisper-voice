"""
Handles interaction with the OpenAI assistant and speech output using edge-tts.
"""
import asyncio
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import edge_tts
import aiohttp

# Import OpenAI client, assistant ID, and initial chat history from settings
from config.settings import openai, ASSISTANT_ID, INITIAL_CHAT_HISTORY, MAX_CHARACTERS

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
    if not user_text.strip():
        print("⚠️ Empty input. Skipping response.")
        return

    print("🧠 Generating response...")
    chat_history.append({"role": "user", "content": user_text})

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=chat_history,
        temperature=0.4,
    )
    assistant_text = response.choices[0].message.content.strip()

    # Trim long replies for voice output
    if len(assistant_text) > MAX_CHARACTERS:
        assistant_text = assistant_text[:MAX_CHARACTERS].rsplit(".", 1)[0] + "."

    chat_history.append({"role": "assistant", "content": assistant_text})
    print("🤖 Assistant:", assistant_text)

    # Speak using edge-tts
    try:
        await speak_with_edge_tts(assistant_text)
    except aiohttp.ClientConnectionError as e:
        print(f"🔌 TTS connection failed: {e}. Skipping speech output.")

async def speak_with_edge_tts(text: str):
    """
    Uses edge-tts to convert text to speech and play it, with retry logic.

    Args:
        text (str): The text to convert and play.
    """
    print("🔊 Generating speech with edge-tts...")
    output_path = "audio/edge_output.mp3"
    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            # Suppress internal Future warnings by gathering the coroutine
            results = await asyncio.gather(
                communicate.save(output_path),
                return_exceptions=True
            )
            if isinstance(results[0], Exception):
                raise results[0]
            break
        except aiohttp.ClientConnectionError as e:
            print(f"🔁 Attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                print("❌ All TTS attempts failed. Skipping speech.")
                return None
            await asyncio.sleep(1)

    # Play audio using PyDub and sounddevice
    audio = AudioSegment.from_file(output_path, format="mp3").normalize().set_frame_rate(24000)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / np.power(2, 15)
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    sd.play(samples, samplerate=audio.frame_rate)
    sd.wait()
    return output_path

def respond_with_text(user_input: str) -> str:
    """
    Generates a text-only response using OpenAI's assistant.

    Args:
        user_input (str): The input message from the user.

    Returns:
        str: Assistant's response text.
    """
    if not user_input.strip():
        return "I'm sorry, I didn't catch that. Could you please repeat?"

    chat_history.append({"role": "user", "content": user_input})

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=chat_history,
        temperature=0.4,
    )

    assistant_text = response.choices[0].message.content.strip()
    chat_history.append({"role": "assistant", "content": assistant_text})
    return assistant_text