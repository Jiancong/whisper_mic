import asyncio
import websockets
import numpy as np
from faster_whisper import WhisperModel
import logging
import time
import requests
import librosa
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters
MODEL_PATH = "medium"  # Whisper model path
OLLAMA_API_URL = "http://localhost:11434/api/chat"  # Ollama API endpoint
TTS_API_URL = "http://localhost:5000/generate"  # TTS service endpoint
SAMPLE_RATE = 16000  # Audio sample rate
TTS_PREDEFINED_AUDIO_DIR = "./tts_predefined_audio"  # Predefined TTS audio directory
TTS_AUDIO_DIR = "./tts_audio"  # Generated TTS audio directory
INTERVIEWER_NAME = "elon_musk"  # Interviewer subdirectory name
MIN_RESPONSE_LENGTH = 100  # Minimum response length (characters)
TIMEOUT_SECONDS = 180  # Timeout duration (seconds, 3 minutes)
MAX_QUESTIONS = 5  # Maximum number of questions
TRANSCRIPTION_INTERVAL = 0.2  # 从0.5秒减少到0.2秒
MIN_AUDIO_BUFFER_SIZE = 3200  # 从8000减少到3200，约0.2秒的音频
MAX_AUDIO_BUFFER_SIZE = SAMPLE_RATE * 10  # 最多保留10秒的音频数据

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

# Preload Whisper model
logger.info("Preloading Whisper model...")
model = WhisperModel(MODEL_PATH, compute_type="float16", device="cuda")
logger.info("Model loaded with GPU support")

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

# Denoise audio
def denoise_audio(audio_data, sample_rate=16000, noise_threshold=0.01):
    try:
        stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
        magnitude, phase = librosa.magphase(stft)
        noise_frames = int(sample_rate * 0.1 / 512)
        noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        mask = magnitude > (noise_profile * noise_threshold)
        magnitude_denoised = magnitude * mask
        stft_denoised = magnitude_denoised * phase
        denoised_audio = librosa.istft(stft_denoised, hop_length=512, length=len(audio_data))
        return denoised_audio.astype(np.float32)
    except Exception as e:
        logger.error(f"Error in denoising: {e}")
        return audio_data

# 完全异步的TTS生成函数
async def generate_tts_audio_async(text, filename, tts_queue):
    """异步生成TTS音频并将结果放入队列"""
    try:
        logger.info(f"异步生成TTS音频: {filename}")
        response = requests.post(
            TTS_API_URL,
            json={
                "ref_audio": "audio_samples/elon-musk-short-sample.wav",
                "ref_text": "So I want to show the people the most importantly, that this is possible. That's the future we could have.",
                "gen_text": text
            }
        )
        response.raise_for_status()
        audio_content = response.content
        
        if not os.path.exists(TTS_AUDIO_DIR):
            os.makedirs(TTS_AUDIO_DIR)
            
        audio_file_path = os.path.join(TTS_AUDIO_DIR, filename)
        with open(audio_file_path, "wb") as f:
            f.write(audio_content)
            
        logger.info(f"TTS audio saved to {audio_file_path}")
        # 将生成的音频文件路径和问题文本放入队列
        await tts_queue.put((audio_file_path, text))
        return audio_file_path
    except Exception as e:
        logger.error(f"Error in async TTS generation: {e}")
        return None

# 保留原有函数以兼容现有代码
async def generate_tts_audio(text, filename):
    try:
        response = requests.post(
            TTS_API_URL,
            json={
                "ref_audio": "audio_samples/elon-musk-short-sample.wav",
                "ref_text": "So I want to show the people the most importantly, that this is possible. That's the future we could have.",
                "gen_text": text
            }
        )
        response.raise_for_status()
        audio_content = response.content
        if not os.path.exists(TTS_AUDIO_DIR):
            os.makedirs(TTS_AUDIO_DIR)
        audio_file_path = os.path.join(TTS_AUDIO_DIR, filename)
        with open(audio_file_path, "wb") as f:
            f.write(audio_content)
        logger.info(f"TTS audio saved to {audio_file_path}")
        return audio_file_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling TTS service: {e}")
        return None

# Send TTS audio to client
async def send_tts_audio(websocket, audio_file_path):
    try:
        with open(audio_file_path, "rb") as f:
            audio_content = f.read()
        await websocket.send(audio_content)
        logger.info(f"Sent audio file: {audio_file_path}")
    except Exception as e:
        logger.error(f"Error sending audio file: {e}")
        await websocket.send(f"Error: Could not send audio file {audio_file_path}")

# 修改生成所有问题的函数为完全异步
async def generate_all_questions_async(conversation_history, start_counter, tts_queue):
    """完全异步生成所有问题音频"""
    question_counter = start_counter
    tasks = []
    
    while question_counter <= MAX_QUESTIONS:
        try:
            next_question = chat_with_ollama(conversation_history + [
                {"role": "user", "content": "Generate the next interview question based on the candidate's response."}
            ])
            
            if next_question:
                filename = f"question_{question_counter}.wav"
                # 创建异步任务但不等待它完成
                task = asyncio.create_task(
                    generate_tts_audio_async(next_question, filename, tts_queue)
                )
                tasks.append(task)
                logger.info(f"创建问题 {question_counter} 的TTS生成任务")
                question_counter += 1
            else:
                logger.error("Failed to generate next question from Ollama API.")
                break
                
        except Exception as e:
            logger.error(f"Error generating question {question_counter}: {e}")
            break
            
    # 返回创建的任务数量，但不等待它们完成
    return question_counter - 1, tasks

# 保留原有函数以兼容现有代码
async def generate_all_questions(conversation_history, start_counter):
    question_counter = start_counter
    while question_counter <= MAX_QUESTIONS:
        try:
            next_question = chat_with_ollama(conversation_history + [{"role": "user", "content": "Generate the next interview question based on the candidate's response."}])
            if next_question:
                filename = f"question_{question_counter}.wav"
                audio_file_path = await generate_tts_audio(next_question, filename)
                if audio_file_path:
                    logger.info(f"Generated audio for question {question_counter}: {next_question}")
                    question_counter += 1
                else:
                    logger.error(f"Failed to generate TTS audio for question {question_counter}")
                    break
            else:
                logger.error("Failed to generate next question from Ollama API.")
                break
        except Exception as e:
            logger.error(f"Error generating question {question_counter}: {e}")
            break
    return question_counter - 1

# File hot reload handler
class HotReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            logger.info(f"Detected change in {event.src_path}, reloading...")

# 实时转录任务 - 优化转录逻辑
async def transcribe_periodically(websocket, audio_buffer, last_transcription_time):
    current_time = time.time()
    # 减少等待时间，只要有足够的音频数据就进行转录
    if (current_time - last_transcription_time >= TRANSCRIPTION_INTERVAL and 
            len(audio_buffer) >= MIN_AUDIO_BUFFER_SIZE):
        try:
            # 减少日志输出频率
            if len(audio_buffer) % 16000 == 0:  # 每秒音频输出一次日志
                logger.info(f"开始转录音频缓冲区，大小: {len(audio_buffer)} 样本")
            
             # 使用最后5秒的音频进行转录，避免处理过长的音频
            last_5_seconds = min(SAMPLE_RATE * 5, len(audio_buffer))
            recent_audio = audio_buffer[-last_5_seconds:]
            
            segments, info = model.transcribe(recent_audio, beam_size=3, language=None)
            transcription = " ".join(segment.text for segment in segments)
            
            if transcription.strip():  # 确保转录内容不为空
                #logger.info(f"检测到语言: {info.language} (置信度: {info.language_probability:.2f})")
                logger.info(f"实时转录: {transcription}")
                # 发送转录文本回客户端
                await websocket.send(f"TRANSCRIPTION: {transcription}")
            else:
                logger.info("转录结果为空")
            return current_time
        except Exception as e:
            logger.error(f"实时转录错误: {e}")
    return last_transcription_time

# Main audio processing function - 修改主处理函数
async def process_audio(websocket, path):
    global conversation_history

    logger.info("Connected to client!")
    audio_buffer = np.array([], dtype=np.float32)
    question_counter = 1
    is_playing_audio = False
    awaiting_response = False
    last_transcription_time = 0
    
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
        await send_tts_audio(websocket, QUESTION_1_FILE)
        logger.info("Sent first question: Can you briefly introduce yourself?")
        is_playing_audio = True
        awaiting_response = True
        question_counter += 1

        # 异步生成后续问题，不阻塞主线程
        _, new_tasks = await generate_all_questions_async(conversation_history, question_counter, tts_queue)
        tts_tasks.extend(new_tasks)
        logger.info(f"已创建 {len(new_tasks)} 个TTS生成任务")

        start_time = time.time()  # Record start time for timeout

        while True:
            try:
                # 添加更详细的日志
                #logger.info("等待客户端消息...")
                message = await asyncio.wait_for(websocket.recv(), timeout=TIMEOUT_SECONDS)
                
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

                    # 添加音频数据统计信息
                    #logger.info(f"收到音频块: {audio_chunk.size} 样本, 最小值: {audio_chunk.min():.4f}, 最大值: {audio_chunk.max():.4f}")
                    
                    # 检查音频数据是否有效
                    if np.all(np.abs(audio_chunk) < 0.003):
                        #logger.warning("音频块可能只包含静音")
                        continue
                    
                    denoised_chunk = denoise_audio(audio_chunk, SAMPLE_RATE)

                    # 限制音频缓冲区大小，防止内存溢出
                    if len(audio_buffer) > MAX_AUDIO_BUFFER_SIZE:
                        # 保留后半部分的音频数据
                        audio_buffer = audio_buffer[-int(MAX_AUDIO_BUFFER_SIZE/2):]
                        logger.info(f"音频缓冲区已达到最大大小，截断至 {len(audio_buffer)} 样本")

                    audio_buffer = np.concatenate((audio_buffer, denoised_chunk))

                    
                    # 实时转录 - 更频繁地进行转录
                    last_transcription_time = await transcribe_periodically(websocket, audio_buffer, last_transcription_time)

                    # 检查是否有足够长的转录内容可以处理
                    if len(audio_buffer) > MIN_AUDIO_BUFFER_SIZE * 4 and awaiting_response:
                        segments, _ = model.transcribe(audio_buffer, beam_size=5)
                        transcription = " ".join(segment.text for segment in segments)
                        logger.info(f"完整转录: {transcription}")
                        
                        # 发送完整转录文本回客户端
                        await websocket.send(f"TRANSCRIPTION: {transcription}")

                        # Check response length
                        if len(transcription) >= MIN_RESPONSE_LENGTH:
                            conversation_history.append({"role": "user", "content": transcription})
                            
                            # 检查队列中是否有准备好的TTS音频
                            try:
                                next_audio_path, question_text = await asyncio.wait_for(tts_queue.get(), timeout=0.1)
                                logger.info(f"使用队列中的下一个问题: {question_text}")
                                await send_tts_audio(websocket, next_audio_path)
                                logger.info(f"Sent next question: {next_audio_path}")
                                is_playing_audio = True
                                awaiting_response = True
                                question_counter += 1
                            except (asyncio.QueueEmpty, asyncio.TimeoutError):
                                # 如果队列为空，检查文件系统
                                next_audio_path = os.path.join(TTS_AUDIO_DIR, f"question_{question_counter}.wav")
                                if os.path.exists(next_audio_path):
                                    await send_tts_audio(websocket, next_audio_path)
                                    logger.info(f"Sent next question from file: {next_audio_path}")
                                    is_playing_audio = True
                                    awaiting_response = True
                                    question_counter += 1
                                else:
                                    logger.info("No more questions available. Ending interview.")
                                    await websocket.send("All questions have been asked. Thank you for the interview!")
                                    break
                                    
                            audio_buffer = np.array([], dtype=np.float32)
                            start_time = time.time()  # Reset timeout after valid response
                        else:
                            logger.info(f"Response too short ({len(transcription)} chars), sending more_details.wav")
                            await send_tts_audio(websocket, MORE_DETAILS_FILE)
                            is_playing_audio = True
                            awaiting_response = True  # Continue awaiting a longer response

                # Check for timeout
                if time.time() - start_time > TIMEOUT_SECONDS:
                    logger.info("Timeout: No sufficient response within 3 minutes.")
                    if os.path.exists(BYE_FILE):
                        await send_tts_audio(websocket, BYE_FILE)
                    await websocket.send("The interview is closed due to inactivity or insufficient response. Thanks!")
                    break

            except asyncio.TimeoutError:
                logger.info("Timeout: No response within 3 minutes.")
                if os.path.exists(BYE_FILE):
                    await send_tts_audio(websocket, BYE_FILE)
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