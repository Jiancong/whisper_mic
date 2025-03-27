import asyncio
import websockets
import numpy as np
from faster_whisper import WhisperModel
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "large"
BUFFER_SIZE = 80000  # 5 seconds at 16000 Hz

async def process_audio(websocket, path):
    logger.info("🔄 Connected to client!")
    try:
        model = WhisperModel(MODEL_PATH, compute_type="float32")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        await websocket.send(f"Error: Failed to load model - {str(e)}")
        return

    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    buffer_index = 0

    try:
        async for message in websocket:
            try:
                audio_chunk = np.frombuffer(message, dtype=np.float32)
                if audio_chunk.size == 0:
                    logger.warning("⚠️ Received empty audio chunk.")
                    continue

                chunk_size = audio_chunk.size
                if buffer_index + chunk_size <= BUFFER_SIZE:
                    audio_buffer[buffer_index:buffer_index + chunk_size] = audio_chunk
                    buffer_index += chunk_size

                if buffer_index >= BUFFER_SIZE:
                    # 移除 samplerate 参数，直接处理 NumPy 数组
                    start_time = time.time()
                    segments, _ = model.transcribe(audio_buffer, language=None)
                    logger.info(f"Transcription took {time.time() - start_time:.2f} seconds")
                    transcription = " ".join(segment.text for segment in segments)
                    await websocket.send(transcription)
                    logger.info(f"🎤 Transcription: {transcription}")
                    buffer_index = 0
            except Exception as e:
                logger.error(f"❌ Error processing audio: {e}", exc_info=True)
                await websocket.send(f"Error: {str(e)}")
    except websockets.exceptions.ConnectionClosed:
        logger.info("❌ Client disconnected unexpectedly.")

start_server = websockets.serve(process_audio, "0.0.0.0", 8765, max_size=10_000_000, ping_interval=20, ping_timeout=60)
logger.info("🚀 Whisper WebSocket server started on ws://localhost:8765")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()