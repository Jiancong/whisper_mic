import asyncio
import websockets
import sounddevice as sd
import numpy as np
import logging
from queue import Queue, Full
import signal
import sys
import io
import wave
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000
CHANNELS = 1
DURATION = None
MAX_MESSAGE_SIZE = 10_000_000  # 10MB to handle larger messages
MAX_QUEUE_SIZE = 200  # 限制队列大小，防止数据积累过多

# 添加日志处理类，用于跟踪重复日志
class ProgressLogger:
    def __init__(self):
        self.last_message = None
        self.repeat_count = 0
        self.last_update_time = 0
        self.progress_chars = ['-', '\\', '|', '/']
        self.progress_index = 0
        
    def log(self, level, message):
        current_time = time.time()
        
        # 如果是新消息或距离上次更新超过1秒
        if message != self.last_message or current_time - self.last_update_time >= 1:
            # 如果有重复消息，先输出重复次数
            if self.repeat_count > 0:
                sys.stdout.write("\r" + " " * 100)  # 清除当前行
                sys.stdout.write(f"\r{self.last_message} (重复 {self.repeat_count} 次)\n")
                sys.stdout.flush()
                self.repeat_count = 0
            
            # 输出新消息
            if level == "INFO":
                logger.info(message)
            elif level == "WARNING":
                logger.warning(message)
            elif level == "ERROR":
                logger.error(message)
                
            self.last_message = message
            self.last_update_time = current_time
        else:
            # 更新进度条
            self.repeat_count += 1
            progress_char = self.progress_chars[self.progress_index]
            self.progress_index = (self.progress_index + 1) % len(self.progress_chars)
            
            sys.stdout.write("\r" + " " * 100)  # 清除当前行
            sys.stdout.write(f"\r{self.last_message} {progress_char} (重复 {self.repeat_count} 次)")
            sys.stdout.flush()

class AudioWebSocketClient:
    def __init__(self):
        self.audio_queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self.running = False
        self.websocket = None
        self.stream = None
        self.last_transcription = ""
        self.progress_logger = ProgressLogger()  # 添加进度日志器

    def audio_callback(self, indata, frames, time, status):
        """Audio callback function for microphone input"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        if self.running:
            audio_data = indata[:, 0].astype(np.float32)
            try:
                # 使用非阻塞方式添加数据，如果队列已满则丢弃
                self.audio_queue.put_nowait(audio_data)
            except Full:
                logger.warning("Audio queue full, dropping frame")

    async def send_audio(self):
        """Send audio from queue to WebSocket"""
        while self.running and self.websocket:
            try:
                # 使用更短的超时时间获取音频数据
                try:
                    audio_data = await asyncio.wait_for(
                        asyncio.to_thread(self.audio_queue.get), 
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                # 添加AUDIO:前缀并发送
                await self.websocket.send(
                    f"AUDIO:{audio_data.tobytes().hex()}"
                    )
                
                self.audio_queue.task_done()
            except Exception as e:
                logger.error(f"Send error: {e}")
                break
    
    async def play_audio(self, audio_data):
        """Play audio from bytes"""
        try:
            # 如果收到的是带前缀的base64数据，需要先解码
            if isinstance(audio_data, str):
                import base64
                audio_data = base64.b64decode(audio_data)
                
            # 将音频数据转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # 播放音频
            sd.play(audio_array, samplerate=SAMPLE_RATE)
            sd.wait()
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    async def receive_and_play_audio(self):
        """Receive data from WebSocket and play audio if it's in bytes"""
        while self.running and self.websocket:
            try:
                data = await self.websocket.recv()

                if isinstance(data, str):
                    if data.startswith("TRANSCRIPTION:"):
                        # 处理转录文本
                        transcription = data[len("TRANSCRIPTION:"):].strip()
                        self.last_transcription = transcription
                        logger.info(f"转录结果: {transcription}")
                    elif data.startswith("AUDIO:"):
                        # 处理音频数据
                        audio_data = data[len("AUDIO:"):]
                        await self.play_audio(audio_data)

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
            callback=self.audio_callback,
            device=None,  # 使用默认设备
            latency='low'  # 使用低延迟设置
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
            ping_interval=30,
            ping_timeout=120,
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