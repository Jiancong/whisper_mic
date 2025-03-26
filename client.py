import asyncio
import websockets
import sounddevice as sd
import numpy as np
import logging
from queue import Queue
import signal
import sys
import io
import wave
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000
CHANNELS = 1
DURATION = None
MAX_MESSAGE_SIZE = 10_000_000  # 10MB to handle larger messages

class AudioWebSocketClient:
    def __init__(self):
        self.audio_queue = Queue()
        self.running = False
        self.websocket = None
        self.stream = None

    def audio_callback(self, indata, frames, time, status):
        """Audio callback function for microphone input"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        if self.running:
            audio_data = indata[:, 0].astype(np.float32)
            self.audio_queue.put(audio_data)

    async def send_audio(self):
        """Send audio from queue to WebSocket"""
        while self.running and self.websocket:
            try:
                audio_data = await asyncio.to_thread(self.audio_queue.get)
                await self.websocket.send(audio_data.tobytes())
                self.audio_queue.task_done()
            except Exception as e:
                logger.error(f"Send error: {e}")
                break

    async def receive_and_play_audio(self):
        """Receive data from WebSocket and play audio if it's in bytes"""
        while self.running and self.websocket:
            try:
                data = await self.websocket.recv()
                if isinstance(data, bytes):
                    # Handle audio data (assumed to be WAV format)
                    with io.BytesIO(data) as wav_io:
                        with wave.open(wav_io, 'rb') as wav_file:
                            sample_rate = wav_file.getframerate()
                            frames = wav_file.readframes(wav_file.getnframes())
                            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                            sd.play(audio_np, samplerate=sample_rate)
                            sd.wait()  # Wait for playback to complete
                else:
                    # Log text data instead of trying to play it
                    logger.info(f"Received text: {data}")
            except Exception as e:
                logger.error(f"Receive error: {e}")
                break

    def start_audio_stream(self):
        """Start the audio input stream"""
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype='float32',
            callback=self.audio_callback
        )
        return self.stream

    def stop(self):
        """Stop the client gracefully"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logger.info("Audio stream stopped")
        if self.websocket:
            asyncio.create_task(self.websocket.close())
            logger.info("WebSocket connection closed")

class HotReloadHandler(FileSystemEventHandler):
    """Handle file changes for hot-reloading"""
    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            logger.info(f"Detected change in {event.src_path}, reloading...")
            # Optional: Add restart logic here, e.g., sys.exit(1) to trigger a restart in a loop

def signal_handler(sig, frame, client):
    """Handle Ctrl+C and other termination signals"""
    logger.info(f"Received signal {sig}, shutting down...")
    client.stop()
    sys.exit(0)

async def test_whisper():
    uri = "ws://localhost:8765"
    client = AudioWebSocketClient()

    # Register signal handlers
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, client))  # Ctrl+C
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, client))  # Unix Ctrl+D
    else:
        signal.signal(signal.SIGBREAK, lambda sig, frame: signal_handler(sig, frame, client))  # Windows Ctrl+Break or Ctrl+Z

    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=60,
            max_size=MAX_MESSAGE_SIZE  # Increase message size limit
        ) as websocket:
            client.websocket = websocket
            logger.info("Connected to Whisper WebSocket Server!")
            
            # Start audio stream
            client.running = True
            with client.start_audio_stream():
                logger.info("Started microphone recording")
                
                # Create send and receive tasks
                send_task = asyncio.create_task(client.send_audio())
                receive_task = asyncio.create_task(client.receive_and_play_audio())
                
                # Wait for tasks to complete
                if DURATION:
                    await asyncio.wait([send_task, receive_task], timeout=DURATION)
                else:
                    await asyncio.gather(send_task, receive_task)
                
    except ConnectionRefusedError:
        logger.error("Failed to connect to server. Is it running?")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        client.stop()

if __name__ == "__main__":
    # Start hot-reloading observer
    event_handler = HotReloadHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=True)
    observer.start()

    try:
        asyncio.run(test_whisper())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Main loop error: {e}")
    finally:
        observer.stop()
        observer.join()