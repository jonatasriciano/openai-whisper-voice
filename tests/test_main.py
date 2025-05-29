import os
from app.config import settings as config

from app.core.conversation import start_conversation

def test_environment_loaded():
    assert config.OPENAI_API_KEY is not None
    assert config.OPENAI_ASSISTANT_ID is not None

def test_audio_directory_exists():
    assert os.path.exists(config.AUDIO_DIR)
    assert os.path.isdir(config.AUDIO_DIR)

def test_output_file_path():
    assert config.OUTPUT_FILE.endswith("input.wav")

def test_conversation_callable():
    assert callable(start_conversation)