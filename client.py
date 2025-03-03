import asyncio
import websockets
import sounddevice as sd
import numpy as np
import logging
from queue import Queue
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BLOCK_SIZE = 8000
CHANNELS = 1
DURATION = None

class AudioWebSocketClient:
    def __init__(self):
        self.audio_queue = Queue()
        self.running = False

    def audio_callback(self, indata, frames, time, status):
        """Audio callback function"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        audio_data = indata[:, 0].astype(np.float32)
        self.audio_queue.put(audio_data)

    async def send_audio(self, websocket):
        """Send audio from queue to websocket"""
        while self.running:
            try:
                audio_data = await asyncio.to_thread(self.audio_queue.get)
                await websocket.send(audio_data.tobytes())
                self.audio_queue.task_done()
            except Exception as e:
                logger.error(f"Send error: {e}")
                break

    async def receive_transcription(self, websocket):
        """Receive transcription from websocket"""
        while self.running:
            try:
                response = await websocket.recv()
                if response.startswith("Error:"):
                    logger.error(f"Server error: {response}")
                else:
                    logger.info(f"Transcription: {response}")
            except Exception as e:
                logger.error(f"Receive error: {e}")
                break

    def start_audio_stream(self):
        """Start the audio input stream"""
        return sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype='float32',
            callback=self.audio_callback
        )

async def test_whisper():
    uri = "ws://localhost:8765"
    client = AudioWebSocketClient()
    
    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=60
        ) as websocket:
            logger.info("✅ Connected to Whisper WebSocket Server!")
            
            # Start audio stream
            client.running = True
            with client.start_audio_stream():
                logger.info("🎙️ Started microphone recording")
                
                # Create tasks for sending and receiving
                send_task = asyncio.create_task(client.send_audio(websocket))
                receive_task = asyncio.create_task(client.receive_transcription(websocket))
                
                # Wait for tasks to complete
                if DURATION:
                    await asyncio.wait([send_task, receive_task], timeout=DURATION)
                else:
                    await asyncio.gather(send_task, receive_task)
                
    except ConnectionRefusedError:
        logger.error("❌ Failed to connect to server. Is it running?")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        client.running = False

if __name__ == "__main__":
    asyncio.run(test_whisper())