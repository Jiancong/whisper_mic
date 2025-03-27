import asyncio
import websockets
import numpy as np
import logging
import time
import requests
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from asr_module import ASRProcessor  # 导入ASR处理器
from tts_module import TTSProcessor  # 导入TTS处理器

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters
OLLAMA_API_URL = "http://localhost:11434/api/chat"  # Ollama API endpoint
TTS_API_URL = "http://localhost:5000/generate"  # TTS service endpoint
SAMPLE_RATE = 16000  # Audio sample rate
TTS_PREDEFINED_AUDIO_DIR = "./tts_predefined_audio"  # Predefined TTS audio directory
TTS_AUDIO_DIR = "./tts_audio"  # Generated TTS audio directory
INTERVIEWER_NAME = "elon_musk"  # Interviewer subdirectory name
MIN_RESPONSE_LENGTH = 20  # Minimum response length (characters)
TIMEOUT_SECONDS = 20  # Timeout duration (seconds, 20 seconds)
MAX_QUESTIONS = 0  # Maximum number of questions
TRANSCRIPTION_INTERVAL = 0.5  # 更频繁地进行转录，从0.5秒减少到0.2秒
MIN_AUDIO_BUFFER_SIZE = 1600  # 约0.1秒的音频
MAX_AUDIO_BUFFER_SIZE = SAMPLE_RATE * 60  # 最多保留60秒的音频数据，从30秒增加到60秒
SILENCE_THRESHOLD = 0.0005  # 静音检测阈值，降低以捕获更多音频

# Predefined audio file paths
QUESTION_1_FILE = os.path.join(TTS_PREDEFINED_AUDIO_DIR, INTERVIEWER_NAME, "question_1.wav")
MORE_DETAILS_FILE = os.path.join(TTS_PREDEFINED_AUDIO_DIR, INTERVIEWER_NAME, "more_details.wav")
BYE_FILE = os.path.join(TTS_PREDEFINED_AUDIO_DIR, INTERVIEWER_NAME, "bye.wav")

# Interview context (resume content)
CONTEXT = """
I'm a Vue.js developer with 5 years of experience, mainly working with Vue 3 and Nuxt.js. 
I have also worked with Bootstrap, Tailwind CSS, and Figma to create clean and responsive designs. 
On the backend side, I have experience with Laravel, which helps me understand API structures better. 
I've built various table structures, including sortable columns, filters, pagination, and pinned columns.

My main skills:
- HTML, CSS, JavaScript
- Vue 2, Vue 3, Composition API
- Tailwind, Bootstrap
- Vue Router, Vuex, Pinia, PrimeVue, VueUse
- API development
- PHP, Laravel
- Vite, Git, etc.
"""

# Initialize conversation history
conversation_history = [
    {"role": "system", "content": "You are an interviewer asking technical questions based on the candidate's resume. Ask one concise question at a time, max 20-30 words, one sentence only."},
    {"role": "user", "content": CONTEXT}
]

# 初始化ASR处理器
asr_processor = ASRProcessor(model_path="medium", compute_type="float16", device="cuda")
logger.info("ASR处理器初始化完成")

# 初始化TTS处理器
tts_processor = TTSProcessor(tts_api_url=TTS_API_URL, tts_audio_dir=TTS_AUDIO_DIR)
logger.info("TTS处理器初始化完成")

# Call Ollama API to generate questions
def chat_with_ollama(messages):
    data = {"model": "phi4:latest", "messages": messages, "stream": False}
    try:
        response = requests.post(OLLAMA_API_URL, json=data)
        response.raise_for_status()
        response_data = response.json()
        if "message" in response_data and "content" in response_data["message"]:
            return response_data["message"]["content"]
        logger.error("Unexpected response format: %s", response_data)
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Ollama API: {e}")
        return None

# File hot reload handler
class HotReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            logger.info(f"Detected change in {event.src_path}, reloading...")

# 实时转录任务 - 使用ASR模块
async def transcribe_periodically(websocket, audio_buffer, last_transcription_time):
    current_time = time.time()
    # 减少等待时间，只要有足够的音频数据就进行转录
    if (current_time - last_transcription_time >= TRANSCRIPTION_INTERVAL and 
            len(audio_buffer) >= MIN_AUDIO_BUFFER_SIZE):
        try:
            # 减少日志输出频率
            if len(audio_buffer) % 32000 == 0:  # 每2秒音频输出一次日志
                logger.info(f"开始转录音频缓冲区，大小: {len(audio_buffer)} 样本")
            
            # 使用ASR模块进行转录 - 增强参数
            transcription, info = asr_processor.transcribe_segment(
                audio_buffer, 
                max_duration_seconds=15,  # 增加处理的音频长度
                beam_size=5  # 增加beam_size提高准确性
            
            )
            
            if transcription.strip():  # 确保转录内容不为空
                logger.info(f"实时转录: {transcription}")
                # 发送转录文本回客户端
                await websocket.send(f"TRANSCRIPTION: {transcription}")
            else:
                logger.debug("转录结果为空")  # 降低日志级别
            return current_time
        except Exception as e:
            logger.error(f"实时转录错误: {e}")
    return last_transcription_time

# Main audio processing function
async def process_audio(websocket, path):
    global conversation_history

    logger.info("Connected to client!")
    audio_buffer = np.array([], dtype=np.float32)
    question_counter = 1
    is_playing_audio = False
    awaiting_response = False
    last_transcription_time = 0
    last_complete_transcription = ""  # 记录上一次完整转录结果
    
    # 创建TTS结果队列
    tts_queue = asyncio.Queue()
    tts_tasks = []

    try:
        # Check if predefined audio directory and files exist
        if not os.path.exists(QUESTION_1_FILE):
            logger.error(f"Predefined audio file not found: {QUESTION_1_FILE}")
            await websocket.send("Error: Predefined question audio file not found.")
            return
        if not os.path.exists(MORE_DETAILS_FILE):
            logger.error(f"Predefined audio file not found: {MORE_DETAILS_FILE}")
            await websocket.send("Error: Predefined more details audio file not found.")
            return
        if not os.path.exists(BYE_FILE):
            logger.error(f"Predefined audio file not found: {BYE_FILE}")
            await websocket.send("Error: Predefined bye audio file not found.")
            return

        # Send the first question audio
        await tts_processor.send_audio(websocket, QUESTION_1_FILE)
        logger.info("Sent first question: Can you briefly introduce yourself?")
        is_playing_audio = True
        awaiting_response = True
        question_counter += 1

        # 添加时间监控代码
        start_gen_time = time.time()
        logger.info(f"开始异步生成问题: {time.strftime('%H:%M:%S', time.localtime(start_gen_time))}")

        # 异步生成后续问题，不阻塞主线程
        _, new_tasks = await tts_processor.generate_all_questions_async(
            chat_with_ollama, conversation_history, question_counter, MAX_QUESTIONS, tts_queue
        )
        
        end_gen_time = time.time()
        gen_duration = end_gen_time - start_gen_time
        logger.info(f"异步问题生成函数返回用时: {gen_duration:.2f}秒")
        logger.info(f"返回时间: {time.strftime('%H:%M:%S', time.localtime(end_gen_time))}")

        tts_tasks.extend(new_tasks)
        logger.info(f"已创建 {len(new_tasks)} 个TTS生成任务")

        start_time = time.time()  # Record start time for timeout
        silence_counter = 0  # 静音计数器
        last_audio_time = time.time()  # 上次接收到有效音频的时间

        while True:
            try:
                # 添加更详细的日志
                message = await asyncio.wait_for(websocket.recv(), timeout=TIMEOUT_SECONDS)

                # logger.info(f"收到客户端消息: {message}")
                
                # 只在收到非二进制数据或首次连接时输出详细日志
                if isinstance(message, str) or audio_buffer.size == 0:
                    logger.info(f"收到客户端消息，类型: {type(message)}")
                
                # Handle client playback status messages
                if isinstance(message, str):
                    if message == "playback_started":
                        is_playing_audio = True
                        logger.info("客户端开始播放音频")
                        continue
                    elif message == "playback_finished":
                        logger.info("客户端完成音频播放")
                        is_playing_audio = False
                        continue

                # Process audio input when no audio is playing
                if not is_playing_audio:
                    # 检查消息是否为二进制数据
                    if not isinstance(message, bytes):
                        logger.warning(f"收到非二进制数据: {message}")
                        continue

                    audio_chunk = np.frombuffer(message, dtype=np.float32)
                    if audio_chunk.size == 0:
                        logger.warning("收到空音频块")
                        continue

                    # # 检测是否为有效音频（非静音）
                    # is_silent = np.mean(np.abs(audio_chunk)) < SILENCE_THRESHOLD
                    
                    # if is_silent:
                    #     silence_counter += 1
                    #     # 每10个静音帧输出一次日志
                    #     if silence_counter % 10 == 0:
                    #         logger.debug(f"检测到静音帧 ({silence_counter})")
                    # else:
                    #     # 重置静音计数器并更新最后有效音频时间
                    #     silence_counter = 0
                    #     last_audio_time = time.time()
                    
                    # 使用ASR模块进行降噪，但保留更多原始信号
                    denoised_chunk = asr_processor.denoise_audio(audio_chunk)

                    # 限制音频缓冲区大小，防止内存溢出
                    if len(audio_buffer) > MAX_AUDIO_BUFFER_SIZE:
                        # 保留后75%的音频数据，丢弃前25%
                        audio_buffer = audio_buffer[-int(MAX_AUDIO_BUFFER_SIZE*0.75):]
                        logger.info(f"音频缓冲区已达到最大大小，截断至 {len(audio_buffer)} 样本")

                    # 将新的音频数据添加到缓冲区
                    audio_buffer = np.concatenate((audio_buffer, denoised_chunk))
                    
                    # 实时转录 - 更频繁地进行转录
                    last_transcription_time = await transcribe_periodically(websocket, audio_buffer, last_transcription_time)

                    # 检查是否有足够长的转录内容可以处理
                    # 当音频缓冲区足够长或者静音持续一段时间后进行完整转录
                    should_transcribe = (
                        len(audio_buffer) > MIN_AUDIO_BUFFER_SIZE * 2 and awaiting_response and
                        (len(audio_buffer) >= SAMPLE_RATE * 3 or  # 至少3秒音频
                         (silence_counter > 5 and time.time() - last_audio_time > 1.0))  # 或1秒静音
                    )
                    
                    if should_transcribe:
                        # 使用ASR模块进行完整转录，增加参数提高准确性
                        transcription, _ = asr_processor.transcribe(
                            audio_buffer, 
                            beam_size=5
                        )
                        
                        # 如果转录结果与上次相同，可能是没有新内容，跳过处理
                        if transcription == last_complete_transcription and len(transcription) > 0:
                            logger.debug("转录结果与上次相同，跳过处理")
                            continue
                            
                        last_complete_transcription = transcription
                        logger.info(f"完整转录: {transcription}")
                        
                        # 发送完整转录文本回客户端
                        await websocket.send(f"TRANSCRIPTION: {transcription}")

                        logger.info(f"len of transcription: {len(transcription)}")

                        # 检查响应长度
                        if len(transcription) >= MIN_RESPONSE_LENGTH:
                            conversation_history.append({"role": "user", "content": transcription})
                            
                            # 检查队列中是否有准备好的TTS音频
                            try:
                                next_audio_path, question_text = await asyncio.wait_for(tts_queue.get(), timeout=0.1)
                                logger.info(f"使用队列中的下一个问题: {question_text}")
                                await tts_processor.send_audio(websocket, next_audio_path)
                                logger.info(f"Sent next question: {next_audio_path}")
                                is_playing_audio = True
                                awaiting_response = True
                                question_counter += 1
                            except (asyncio.QueueEmpty, asyncio.TimeoutError):
                                # 如果队列为空，检查文件系统
                                next_audio_path = os.path.join(TTS_AUDIO_DIR, f"question_{question_counter}.wav")
                                if os.path.exists(next_audio_path):
                                    await tts_processor.send_audio(websocket, next_audio_path)
                                    logger.info(f"Sent next question from file: {next_audio_path}")
                                    is_playing_audio = True
                                    awaiting_response = True
                                    question_counter += 1
                                else:
                                    logger.info("No more questions available. Ending interview.")
                                    await websocket.send("All questions have been asked. Thank you for the interview!")
                                    break
                                    
                            # 清空音频缓冲区，准备接收下一个回答
                            audio_buffer = np.array([], dtype=np.float32)
                            start_time = time.time()  # Reset timeout after valid response
                            silence_counter = 0
                        else:
                            # 不要立即发送more_details，给用户更多时间继续说话
                            if len(transcription) > 0 and len(audio_buffer) < SAMPLE_RATE * 10:  # 如果有内容但不够长，且音频不超过10秒，继续等待
                                continue
                            
                            logger.info(f"Response too short ({len(transcription)} chars), sending more_details.wav")
                            await tts_processor.send_audio(websocket, MORE_DETAILS_FILE)
                            is_playing_audio = True
                            awaiting_response = True  # Continue awaiting a longer response
                            # 不清空音频缓冲区，保留已收集的音频

                # Check for timeout
                if time.time() - start_time > TIMEOUT_SECONDS:
                    logger.info("Timeout 111 : No sufficient response within 20 seconds.")
                    if os.path.exists(BYE_FILE):
                        await tts_processor.send_audio(websocket, BYE_FILE)
                    await websocket.send("The interview is closed due to inactivity or insufficient response. Thanks!")
                    break

            except asyncio.TimeoutError:
                logger.info("Timeout 222: No response within 20 seconds.")
                if os.path.exists(BYE_FILE):
                    await tts_processor.send_audio(websocket, BYE_FILE)
                await websocket.send("The interview is closed due to inactivity. Thanks for your time!")
                break

            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}", exc_info=True)
                await websocket.send(f"Error occurred: {str(e)}")

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected unexpectedly.")
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        await websocket.send(f"Error: {str(e)}")
    finally:
        # 取消所有未完成的TTS任务
        for task in tts_tasks:
            if not task.done():
                task.cancel()

# Main execution block
if __name__ == "__main__":
    # Set up file hot reload observer
    event_handler = HotReloadHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=True)
    observer.start()

    # Start WebSocket server
    start_server = websockets.serve(process_audio, "0.0.0.0", 8765, max_size=10_000_000, ping_interval=30, ping_timeout=120)
    logger.info("Whisper WebSocket server started on ws://localhost:8765")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

    # Clean up observer
    observer.stop()
    observer.join()