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
import os
import soundfile as sf
from audio_sender import AudioSender  # 导入新的AudioSender类

# Configure logging
logging.basicConfig(level=logging.INFO)

logging.getLogger('audio_sender').setLevel(logging.INFO)  # 设置为INFO级别，以确保所有日志都被记录
logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000
BLOCK_SIZE = 80000  # 增加块大小，从4000到80000，每次收集5秒的音频
CHANNELS = 1
DURATION = None
MAX_MESSAGE_SIZE = 10_000_000  # 10MB to handle larger messages
MAX_QUEUE_SIZE = 1000  # 增加队列大小，从200到400，可以缓存更多音频
SILENCE_THRESHOLD = 0.1  # 静音检测阈值
SILENCE_DURATION = 5  # 静音持续5秒后停止录音

# 添加调试音频保存的全局计数器
debug_counter = 0

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
        
        # 创建音频发送器，添加VAD参数
        self.audio_sender = AudioSender(
            sample_rate=SAMPLE_RATE, 
            debug_dir="client_debug_audio",
            vad_threshold=0.015,  # 稍微提高VAD阈值，更好地过滤低能量噪音
            min_speech_duration=0.3  # 最小语音段长度为0.3秒
        )
        
        # 记录音频回调的统计信息
        self.callback_count = 0

        # 添加静音检测相关变量
        self.last_active_time = time.time()  # 上次检测到有效音频的时间
        self.is_silent = False  # 当前是否处于静音状态
        self.paused = False  # 录音是否已暂停
        self.consecutive_silence_blocks = 0  # 连续静音块计数

    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback function for microphone input"""
        if status:
            logger.warning(f"Audio callback status: {status}")

        if self.running:
            # 获取音频数据
            audio_data = indata[:, 0].astype(np.float32)

            # 记录回调次数
            self.callback_count += 1
            
            # 记录音频数据的统计信息
            max_val = np.max(np.abs(audio_data))
            
            # 检测是否为静音
            is_current_block_silent = max_val < SILENCE_THRESHOLD

            # 提高日志级别，确保我们能看到每次回调的信息
            logger.info(f"音频回调 #{self.callback_count}: 收到 {len(audio_data)} 样本, 最大值: {max_val:.6f}, 静音: {is_current_block_silent}")
            
            # 只有在非暂停状态下才保存调试音频
            if not self.paused:
                # 保存回调接收到的音频数据，增加保存频率
                self.audio_sender.save_debug_audio(audio_data, prefix="callback")

            # 静音检测逻辑
            if is_current_block_silent:
                self.consecutive_silence_blocks += 1
                logger.info(f"检测到静音块 #{self.consecutive_silence_blocks}")
                
                # 如果连续静音时间超过阈值且录音未暂停
                if self.consecutive_silence_blocks * (BLOCK_SIZE / SAMPLE_RATE) >= SILENCE_DURATION and not self.paused:
                    logger.info(f"检测到连续静音超过 {SILENCE_DURATION} 秒，暂停录音")
                    self.paused = True
                    # 发送一个特殊信号到队列，表示用户停止说话
                    try:
                        # 创建一个特殊的静音标记数据（非常小的值）
                        silence_marker = np.zeros(1, dtype=np.float32)
                        self.audio_queue.put_nowait(silence_marker)
                        logger.info("已添加静音标记到队列")
                    except Full:
                        logger.warning("音频队列已满，无法添加静音标记")
            else:
                # 如果当前块不是静音
                if self.paused:
                    logger.info("检测到新的语音输入，恢复录音")
                    self.paused = False            

                # 重置连续静音计数
                self.consecutive_silence_blocks = 0
                self.last_active_time = time.time()  # 使用导入的time模块，而不是参数                  
            
            # 只有在非暂停状态或者刚检测到静音开始时才添加数据到队列
            if not self.paused or (is_current_block_silent and self.consecutive_silence_blocks == 1):
                try:
                    # 使用非阻塞方式添加数据，如果队列已满则丢弃
                    self.audio_queue.put_nowait(audio_data)
                    logger.info(f"已添加音频数据到队列: {len(audio_data)} 样本, 队列大小: {self.audio_queue.qsize()}/{self.audio_queue.maxsize}")
                    
                    # 检查队列大小是否超过阈值
                    if self.audio_queue.qsize() > 10 and self.audio_queue.qsize() % 5 == 0:
                        logger.warning(f"队列积累了大量数据: {self.audio_queue.qsize()} 项，可能存在处理延迟")
                except Full:
                    logger.warning("音频队列已满，丢弃一帧")

    async def send_audio(self):
        """Send audio from queue to WebSocket"""

        logger.info("开始发送音频   send_audio")
        # 设置WebSocket连接
        self.audio_sender.set_websocket(self.websocket)
        
        # 记录开始处理音频队列的时间
        start_time = time.time()
        logger.info(f"开始处理音频队列，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"队列初始状态: {self.audio_queue.qsize()}/{self.audio_queue.maxsize}")

        # 添加一个标志，表示是否已经启动了处理任务
        self._audio_processing_active = True

        # 处理音频队列
        
        try:
            # 使用AudioSender处理音频队列
            await self.audio_sender.process_audio_queue(self.audio_queue, self.running)
        except Exception as e:
            logger.error(f"处理音频队列时发生异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
            # 如果发生异常，等待一段时间后重试
            if self.running and self._audio_processing_active:
                logger.info("尝试重新启动音频队列处理...")
                self._audio_processing_active = False  # 防止重复启动
                await asyncio.sleep(1)
                asyncio.create_task(self.send_audio())  # 使用create_task而不是递归调用
                return
        finally:
            self._audio_processing_active = False

        # 记录结束处理音频队列的时间
        end_time = time.time()
        logger.info(f"结束处理音频队列，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}, 总时长: {end_time - start_time:.2f}秒")

    async def receive_and_play_audio(self):
        """Receive data from WebSocket and play audio if it's in bytes"""
        while self.running and self.websocket:
            try:
                data = await self.websocket.recv()
                if isinstance(data, bytes):
                    # 通知服务器开始播放音频
                    await self.websocket.send("playback_started")
                    print("playback_started 开始播放从服务端过来的音频")
                    
                    # Handle audio data (assumed to be WAV format)
                    with io.BytesIO(data) as wav_io:
                        with wave.open(wav_io, 'rb') as wav_file:
                            sample_rate = wav_file.getframerate()
                            frames = wav_file.readframes(wav_file.getnframes())
                            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                            sd.play(audio_np, samplerate=sample_rate)
                            sd.wait()  # Wait for playback to complete
                    
                    # 通知服务器音频播放完成
                    await self.websocket.send("playback_finished")
                    print("playback_finished  播放服务端音频完毕")
                else:
                    print("接收到文本数据")
                    # 检查是否是转录文本
                    if isinstance(data, str) and data.startswith("TRANSCRIPTION:"):
                        logger.info("收到了从服务端发来的转录文本")
                        transcription = data[14:].strip()  # 去掉前缀
                        if transcription != self.last_transcription:  # 避免重复显示相同的转录
                            self.last_transcription = transcription
                            logger.info(f"转录文本内容为: {transcription}")
                    else:
                        # Log text data instead of trying to play it
                        logger.info(f"Received text: {data}")
            except Exception as e:
                logger.error(f"Receive error: {e}")
                break

    def start_audio_stream(self):
        """Start the audio input stream"""
        # 列出可用的音频设备
        devices = sd.query_devices()
        logger.info(f"可用的音频设备: {devices}")
        
        # 获取默认输入设备
        default_input = sd.query_devices(kind='input')
        logger.info(f"默认输入设备: {default_input}")
        
        # 尝试使用WASAPI接口，通常提供更好的性能
        wasapi_devices = []
        device_index = None
        
        for i, d in enumerate(devices):
            if isinstance(d, dict) and 'name' in d and 'WASAPI' in str(d['name']) and d.get('max_input_channels', 0) > 0:
                wasapi_devices.append((i, d))
        
        # 如果找到WASAPI设备，使用第一个
        if wasapi_devices:
            device_index, device = wasapi_devices[0]
            logger.info(f"使用WASAPI设备: {device['name']}, 索引: {device_index}")
        
        # 使用选定的输入设备
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype='float32',
            callback=self.audio_callback,
            device=device_index,  # 使用选定的设备或默认设备
            latency='low'  # 使用低延迟设置
        )
        
        logger.info(f"已启动音频流: 采样率={SAMPLE_RATE}, 块大小={BLOCK_SIZE}, 通道数={CHANNELS}, 设备索引={device_index}")
        return self.stream

    def stop(self):
        """Stop the client gracefully"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logger.info("Audio stream stopped")
        if self.websocket and self.websocket.open:
            # 不使用asyncio.create_task，因为它可能不会被执行
            try:
                # 使用非阻塞方式关闭WebSocket
                self.websocket.close_code = 1000
                self.websocket.close_reason = "Client shutting down"
                self.websocket.close_connection_task = None
                logger.info("WebSocket connection marked for closing")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")

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
    # 强制退出程序，不等待其他任务
    os._exit(0)  # 使用os._exit强制退出，而不是sys.exit

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
            # 设置音频发送器的WebSocket连接
            client.audio_sender.set_websocket(websocket)
            
            logger.info("Connected to Whisper WebSocket Server!")
            
            # Start audio stream
            client.running = True
            with client.start_audio_stream():
                logger.info("Started microphone recording")
                
                # Create send and receive tasks
                send_task = asyncio.create_task(client.send_audio())
                receive_task = asyncio.create_task(client.receive_and_play_audio())
                
                # Wait for tasks to complete
                try:
                    if DURATION:
                        await asyncio.wait([send_task, receive_task], timeout=DURATION)
                    else:
                        await asyncio.gather(send_task, receive_task)
                except asyncio.CancelledError:
                    # 正确处理任务取消
                    logger.info("Tasks cancelled, cleaning up...")
                finally:
                    # 确保任务被取消
                    send_task.cancel()
                    receive_task.cancel()
                    # 等待任务取消完成
                    try:
                        await asyncio.wait([send_task, receive_task], timeout=2)
                    except Exception:
                        pass
                
    except ConnectionRefusedError:
        logger.error("Failed to connect to server. Is it running?")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        client.stop()
        # 确保所有异步任务都被取消
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

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
        # 强制退出
        os._exit(0)
    except Exception as e:
        logger.error(f"Main loop error: {e}")
        # 强制退出
        os._exit(1)
    finally:
        observer.stop()
        observer.join()