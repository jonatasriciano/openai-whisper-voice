# Makefile for openai-whisper-voice assistant

# Run the assistant locally
run:
	. venv/bin/activate && python app/main.py

# Create a virtual environment and install dependencies
init:
	rm -rf venv && python -m venv venv && . venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Run tests
test:
	pytest tests/

# Build the Docker image
docker-build:
	docker build -t whisper-assistant .

# Run the Docker container using docker-compose
docker-up:
	docker compose up --build

# Remove .pyc and __pycache__ files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +; find . -name "*.pyc" -delete

# Autoformat Python code
format:
	black app/ tests/