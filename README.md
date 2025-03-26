# whisper_mic

This project is based on several open-source projects to implement ASR methods for real-time microphone input.

## Installation

Install the rquirements.txt for all dpendencies.

```bash
pip install -r requirements.txt
```

## Run

```bash
python server.py
```

## Usage

### ASR

We use a Whisper model for ASR. It runs at  ### ASR

Needs to activate the whisper environment in another command line. Then run the app.py in that environment.

```bash
conda activate python12env
python server.py

We use a Whisper model for ASR. It runs at  URL_ADDRESS:5000/transcribe

### LLM with Ollama
We use a Ollama service for LLM inference. Make sure Ollama is running with the required LLM instance. 
It runs at  http://localhost:11434/api/chat

### TTS

We use F5-TTS-main for TTS. It runs at  for TTS. It runs at  URL_ADDRESS:5000/generate

This is needs to activate the f5-tts environment in another command line. Then run the app.py in that environment.

```bash
conda activate python310
python app.py
```


