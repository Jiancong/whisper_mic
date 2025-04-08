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
OLLAMA_LLM_NAME= "qwen2.5-coder:14b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"  # Ollama API endpoint
TTS_API_URL = "http://localhost:5000/generate"  # TTS service endpoint
SAMPLE_RATE = 16000  # Audio sample rate
TTS_PREDEFINED_AUDIO_DIR = "./tts_predefined_audio"  # Predefined TTS audio directory
TTS_AUDIO_DIR = "./tts_audio"  # Generated TTS audio directory
INTERVIEWER_NAME = "elon_musk"  # Interviewer subdirectory name
MIN_RESPONSE_LENGTH = 20  # Minimum response length (characters)
TIMEOUT_SECONDS = 60  # 增加超时时间从20秒到60秒
MAX_QUESTIONS = 0  # Maximum number of questions
TRANSCRIPTION_INTERVAL = 3  # 增加转录间隔时间，从2秒到5秒
MIN_AUDIO_BUFFER_SIZE = SAMPLE_RATE * 3  # 至少需要5秒的音频才开始转录，而不是0.1秒
MAX_AUDIO_BUFFER_SIZE = SAMPLE_RATE * 120  # 最多保留120秒的音频数据，从60秒增加到120秒
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
    {
        "role": "system", 
        "content": "You are an interviewer asking technical questions based on the candidate's resume. Ask one concise question at a time, max 10-20 words, one sentence only."
    },
    {
        "role": "user", 
        "content": CONTEXT
    }
]

# 初始化ASR处理器
asr_processor = ASRProcessor(model_path="medium", compute_type="float16", device="cuda")
logger.info("ASR处理器初始化完成")

# 初始化TTS处理器
tts_processor = TTSProcessor(tts_api_url=TTS_API_URL, tts_audio_dir=TTS_AUDIO_DIR)
logger.info("TTS处理器初始化完成")

# Call Ollama local API to generate questions
def chat_with_ollama(messages):
    data = {
        "model": OLLAMA_LLM_NAME, 
        "messages": messages, 
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=data)
        response.raise_for_status()
        response_data = response.json()
        if "message" in response_data and "content" in response_data["message"]:
            content = response_data["message"]["content"]
            # 添加日志记录生成的问题
            logger.info(f"Ollama生成的问题: {content}")
            return content
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
    # 只有当累积了足够长的音频(至少5秒)且距离上次转录已经过了足够时间，才进行转录
    if (current_time - last_transcription_time >= TRANSCRIPTION_INTERVAL and 
            len(audio_buffer) >= SAMPLE_RATE * 5):    # 至少5秒音频才尝试转录
        try:
            logger.info(f"开始转录音频缓冲区，大小: {len(audio_buffer)} 样本，最大值: {np.max(np.abs(audio_buffer))}, 时长: {len(audio_buffer)/SAMPLE_RATE:.2f}秒")
            
            # 保存当前要转录的完整音频缓冲区
            asr_processor.save_debug_audio(audio_buffer, prefix="periodic_transcribe_buffer")            
            
            # 确保音频数据格式正确
            if audio_buffer.dtype != np.float32:
                logger.warning(f"音频数据类型不是float32，而是{audio_buffer.dtype}，尝试转换")
                audio_buffer = audio_buffer.astype(np.float32)

            # 确保音频数据幅度在[-1, 1]范围内
            max_val = np.max(np.abs(audio_buffer))
            if max_val > 1.0:
                logger.warning(f"音频数据超出范围，最大值为{max_val}，进行归一化")
                audio_buffer = audio_buffer / max_val                

           
            # 不使用降噪逻辑
            transcription, info = asr_processor.transcribe(
                audio_buffer, 
                beam_size=5  # 增加beam_size提高准确性
            )
            
            if transcription.strip():  # 确保转录内容不为空
                logger.info(f"实时转录: {transcription}")
                # 发送转录文本回客户端
                await websocket.send(f"TRANSCRIPTION: {transcription}")
            else:
                logger.debug("转录结果为空，可能是音频质量问题或背景噪音")  # 降低日志级别
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

        awaiting_response = True
        question_counter += 1

        # 添加时间监控代码
        start_gen_time = time.time()
        logger.info(f"开始异步生成问题: {time.strftime('%H:%M:%S', time.localtime(start_gen_time))}")

        # 异步生成后续问题，不阻塞主线程
        questions, new_tasks = await tts_processor.generate_all_questions_async(
            chat_with_ollama, 
            conversation_history, 
            question_counter, 
            MAX_QUESTIONS, 
            tts_queue
        )
        
        # 添加日志记录所有生成的问题
        if questions:
            logger.info(f"异步生成的所有问题: {questions}")
        
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

                # 重置超时计时器
                start_time = time.time()
                # 记录当前超时设置
                logger.debug(f"当前超时设置: {TIMEOUT_SECONDS}秒, 剩余时间: {TIMEOUT_SECONDS - (time.time() - start_time):.2f}秒")

                # logger.info(f"收到客户端消息: {message}")
                
                # 只在收到非二进制数据或首次连接时输出详细日志
                if isinstance(message, str) or audio_buffer.size == 0:
                    logger.info(f"收到客户端消息，类型: {type(message)}")
                
                # 处理客户端发送的停止说话信号
                if message == "USER_STOPPED_SPEAKING" or message == "SILENCE_DETECTED":
                    logger.info(f"收到{message}信号，进行最终转录")
                    
                    # 如果缓冲区有足够的音频数据，进行最终转录
                    if len(audio_buffer) >= MIN_AUDIO_BUFFER_SIZE and awaiting_response:
                        # 保存最终的音频缓冲区用于调试
                        asr_processor.save_debug_audio(audio_buffer, prefix="final_transcribe_buffer")
                        
                        # 进行最终转录
                        transcription, _ = asr_processor.transcribe(audio_buffer, beam_size=5)
                        
                        if transcription and len(transcription) > 0:
                            logger.info(f"最终转录结果: {transcription}")
                            await websocket.send(f"TRANSCRIPTION: {transcription}")
                            
                            # 如果转录内容足够长，处理用户回答
                            if len(transcription) >= MIN_RESPONSE_LENGTH:
                                conversation_history.append({"role": "user", "content": transcription})
                                
                                # 检查队列中是否有准备好的TTS音频
                                try:
                                    next_audio_path, question_text = await asyncio.wait_for(tts_queue.get(), timeout=0.1)
                                    logger.info(f"使用队列中的下一个问题: {question_text}")
                                    await tts_processor.send_audio(websocket, next_audio_path)
                                    logger.info(f"Sent next question: {next_audio_path}")

                                    awaiting_response = True
                                    question_counter += 1
                                except (asyncio.QueueEmpty, asyncio.TimeoutError):
                                    # 如果队列为空，检查文件系统
                                    next_audio_path = os.path.join(TTS_AUDIO_DIR, f"question_{question_counter}.wav")
                                    if os.path.exists(next_audio_path):
                                        await tts_processor.send_audio(websocket, next_audio_path)
                                        logger.info(f"Sent next question from file: {next_audio_path}")

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
                                logger.info(f"最终转录结果太短 ({len(transcription)} 字符)，发送more_details.wav")
                                await tts_processor.send_audio(websocket, MORE_DETAILS_FILE)
                                awaiting_response = True
                                # 清空音频缓冲区，准备接收新的回答
                                audio_buffer = np.array([], dtype=np.float32)
                        else:
                            logger.info("最终转录结果为空，继续等待用户输入")
                    
                    continue
                
                # Handle client playback status messages
                if isinstance(message, str) and message.startswith("AUDIO:"):
                    # 处理带AUDIO:前缀的音频数据
                    try:
                        # 客户端发送的是二进制数据，需要先解码
                        audio_bytes = message[6:].encode('latin1')  # 使用latin1编码将字符串转回二进制
                        audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                        logger.info(f"收到客户端音频数据，大小: {len(audio_chunk)} 样本，最大值: {np.max(np.abs(audio_chunk))}")

                        # 保存调试音频
                        asr_processor.save_debug_audio(audio_chunk, prefix="received_audio")
                            
                        # 将新的音频数据添加到缓冲区
                        audio_buffer = np.concatenate((audio_buffer, audio_chunk))
                        logger.info(f"音频缓冲区当前大小: {len(audio_buffer)} 样本, 时长: {len(audio_buffer)/SAMPLE_RATE:.2f}秒")
                        
                        # 限制缓冲区大小
                        if len(audio_buffer) > MAX_AUDIO_BUFFER_SIZE:
                            audio_buffer = audio_buffer[-MAX_AUDIO_BUFFER_SIZE:]


                        # 重置超时计时器，只要收到音频数据就刷新
                        start_time = time.time()
                        silence_counter = 0  # 重置静音计数器

                        # 定期进行转录
                        last_transcription_time = await transcribe_periodically(websocket, audio_buffer, last_transcription_time)
                        
                    except Exception as e:
                        logger.error(f"处理音频数据错误: {e}")
                    continue
                elif message == "playback_started":
                    is_playing_audio = True
                    logger.info("客户端开始播放从服务端发送过去的音频")
                    continue
                elif message == "playback_finished":
                    logger.info("客户端完成从服务端发功过去的音频播放")
                    is_playing_audio = False
                    continue

                # Process audio input when no audio is playing
                if not is_playing_audio:
                    # 检查消息是否为二进制数据
                    if not isinstance(message, bytes):
                        # 修改这里，不再简单地跳过非二进制数据
                        if message != "SILENCE_DETECTED":  # 已经在上面处理了SILENCE_DETECTED
                            logger.warning(f"收到非二进制数据: {message} 。我们期待的是二进制音频数据，跳过。")
                        continue

                    logger.info(f"收到客户端音频数据，大小: {len(message)} 字节")
                    audio_chunk = np.frombuffer(message, dtype=np.float32)
                    if audio_chunk.size == 0:
                        logger.warning("收到空音频块, 跳过重听")
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
                    
                    #logger.info("开始降噪")
                    # 使用ASR模块进行降噪，但保留更多原始信号
                    #denoised_chunk = asr_processor.denoise_audio(audio_chunk)
                    denoised_chunk = audio_chunk
                    #logger.info("降噪完成")

                    # 限制音频缓冲区大小，防止内存溢出
                    if len(audio_buffer) > MAX_AUDIO_BUFFER_SIZE:
                        # 保留后75%的音频数据，丢弃前25%
                        audio_buffer = audio_buffer[-int(MAX_AUDIO_BUFFER_SIZE*0.75):]
                        logger.info(f"音频缓冲区已达到最大大小，截断至 {len(audio_buffer)} 样本")

                    # 将新的音频数据添加到缓冲区
                    audio_buffer = np.concatenate((audio_buffer, denoised_chunk))
                    
                    logger.info("开始实时转录操作并检查转录内容是否够长")
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
                        logger.info("应该进行转录")
                        last_transcription_time = time.time()
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

                                awaiting_response = True
                                question_counter += 1
                            except (asyncio.QueueEmpty, asyncio.TimeoutError):
                                # 如果队列为空，检查文件系统
                                next_audio_path = os.path.join(TTS_AUDIO_DIR, f"question_{question_counter}.wav")
                                if os.path.exists(next_audio_path):
                                    await tts_processor.send_audio(websocket, next_audio_path)
                                    logger.info(f"Sent next question from file: {next_audio_path}")

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
                elapsed = time.time() - start_time
                logger.warning(f"超时: {elapsed:.2f}秒内没有收到响应 (设置的超时时间: {TIMEOUT_SECONDS}秒)")

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