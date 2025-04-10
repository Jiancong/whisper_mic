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
OLLAMA_LLM_NAME= "gemma3:12b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"  # Ollama API endpoint
TTS_API_URL = "http://localhost:5000/generate"  # TTS service endpoint
QUESTION_GEN_API_URL = "http://localhost:5001/questions_status"  # 问题生成服务API

SAMPLE_RATE = 16000  # Audio sample rate
TTS_PREDEFINED_AUDIO_DIR = "tts_predefined_audio"  # Predefined TTS audio directory
TTS_AUDIO_DIR = "tts_audio"  # Generated TTS audio directory
INTERVIEWER_NAME = "elon_musk"  # Interviewer subdirectory name
MIN_RESPONSE_LENGTH = 20  # Minimum response length (characters)
TIMEOUT_SECONDS = 60  # 增加超时时间从20秒到60秒
MAX_QUESTIONS = 2  # Maximum number of questions
TRANSCRIPTION_INTERVAL = 3  # 增加转录间隔时间，从2秒到5秒
MIN_AUDIO_BUFFER_SIZE = SAMPLE_RATE * 3  # 至少需要5秒的音频才开始转录，而不是0.1秒
MAX_AUDIO_BUFFER_SIZE = SAMPLE_RATE * 120  # 最多保留120秒的音频数据，从60秒增加到120秒
SILENCE_THRESHOLD = 0.0005  # 静音检测阈值，降低以捕获更多音频

QUESTIONS_CHECK_INTERVAL = 2  # 检查问题生成状态的间隔（秒）
MAX_WAIT_TIME = 300  # 最长等待问题生成的时间（秒）

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
        "content": "You are an interviewer asking technical questions based on the candidate's resume. Ask one concise question at a time, max 10 words, one sentence only."
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

# 在文件顶部添加一个新的常量
SILENCE_ANALYSIS_THRESHOLD = 5  # 静音5秒后分析用户回答状态

# 检查问题生成状态
async def check_questions_ready(tts_queue):
    """检查问题是否已生成完毕"""
    try:
        # 首先检查本地文件，第1个问题放在预定目录，不需要检测
        if MAX_QUESTIONS > 1 and tts_processor.check_questions_ready(TTS_AUDIO_DIR, MAX_QUESTIONS + 1):
            logger.info("本地文件已准备就绪，将使用本地文件")
            return True
        elif MAX_QUESTIONS <= 1 and os.path.exists(QUESTION_1_FILE) :
            logger.info("本地文件已准备就绪，将使用本地文件")
            return True
        else:
            logger.info("本地文件不完整，将尝试通过API检查")


        question_counter = 2
        while question_counter <= MAX_QUESTIONS:
            audio_file_path = os.path.join(TTS_AUDIO_DIR, f"question_{question_counter}.wav")
            if not os.path.exists(audio_file_path):
                logger.warning(f"问题文件不存在: {audio_file_path}")
                return False
            
            # 异步生成后续问题，不阻塞主线程
            questions, new_tasks = await tts_processor.generate_all_questions_async(
                chat_with_ollama, 
                conversation_history, 
                question_counter, 
                MAX_QUESTIONS, 
                tts_queue
            )       
            question_counter += 1    

    except Exception as e:
        logger.error(f"检查问题状态时出错: {e}")
        return False

# 等待问题生成完毕
async def wait_for_questions(websocket, tts_queue):
    """等待问题生成完毕，并通知客户端进度"""
    start_time = time.time()
    
    # 发送初始状态消息
    await websocket.send("STATUS: 正在准备面试问题，请稍候...")
    
    while True:
        # 检查是否超时
        if time.time() - start_time > MAX_WAIT_TIME:
            logger.warning(f"等待问题生成超时 ({MAX_WAIT_TIME}秒)")
            await websocket.send("STATUS: 问题生成超时，将使用已有问题继续")
            return False
            
        # 检查问题是否已生成完毕
        if await check_questions_ready(tts_queue):
            await websocket.send("STATUS: 面试问题已准备就绪，即将开始面试")
            return True
            
        # 获取当前已生成的问题数量
        current_count = tts_processor.count_existing_questions(TTS_AUDIO_DIR)
        total_count = MAX_QUESTIONS + 1
        
        # 发送进度消息
        progress = min(100, int(current_count / total_count * 100))
        await websocket.send(f"STATUS: 正在准备面试问题 ({current_count}/{total_count})... {progress}%")
        
        # 等待一段时间再检查
        await asyncio.sleep(QUESTIONS_CHECK_INTERVAL)

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
    # 创建TTS结果队列
    tts_queue = asyncio.Queue()
    tts_tasks = []

    audio_buffer = np.array([], dtype=np.float32)
    question_counter = 1
    is_playing_audio = False
    awaiting_response = False
    last_transcription_time = 0
    last_complete_transcription = ""  # 记录上一次完整转录结果

    # 添加新变量
    current_question = "Can you briefly introduce yourself?"  # 默认第一个问题
    last_silence_analysis_time = 0  # 上次分析用户回答状态的时间
    silence_start_time = None  # 静音开始时间
    is_silence = False  # 当前是否处于静音状态
    
    # 添加新变量，用于跟踪连续静音检测次数和是否已发送更多细节提示
    consecutive_silence_count = 0  # 连续静音检测次数
    more_details_sent = False  # 是否已发送更多细节提示
    current_question_id = 1  # 当前问题ID，用于跟踪问题变化

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
                
                # 只在收到非二进制数据或首次连接时输出详细日志
                if isinstance(message, str) or audio_buffer.size == 0:
                    logger.info(f"收到客户端消息，类型: {type(message)}")
                
                # 处理客户端发送的停止说话信号
                if message == "USER_STOPPED_SPEAKING" or message == "SILENCE_DETECTED":
                    logger.info(f"收到{message}信号，进行最终转录")
                    
                    # 如果收到SILENCE_DETECTED信号，增加连续静音计数
                    if message == "SILENCE_DETECTED" and more_details_sent:
                        consecutive_silence_count += 1
                        logger.info(f"连续静音计数: {consecutive_silence_count}/3")
                        
                        # 如果连续三次检测到静音，且已经发送过更多细节提示，则认为用户已完成回答
                        if consecutive_silence_count >= 3:
                            logger.info("连续三次检测到静音，认为用户已完成回答，准备进入下一个问题")
                            
                            # 进行最终转录
                            if len(audio_buffer) >= MIN_AUDIO_BUFFER_SIZE:
                                transcription, _ = asr_processor.transcribe(audio_buffer, beam_size=5)
                                
                                if transcription and len(transcription) > 0:
                                    logger.info(f"最终转录结果: {transcription}")
                                    await websocket.send(f"TRANSCRIPTION: {transcription}")
                                    
                                    # 将用户回答添加到对话历史
                                    conversation_history.append({"role": "user", "content": transcription})
                            
                            # 重置连续静音计数和更多细节标志
                            consecutive_silence_count = 0
                            more_details_sent = False
                            
                            # 生成下一个问题
                            if question_counter <= MAX_QUESTIONS:
                                logger.info("根据用户回答生成下一个问题...")
                                
                                # 生成下一个问题
                                next_question = chat_with_ollama(conversation_history)
                                
                                if next_question:
                                    logger.info(f"生成的下一个问题: {next_question}")
                                    
                                    # 添加到对话历史
                                    conversation_history.append({"role": "assistant", "content": next_question})
                                    current_question = next_question
                                    current_question_id = question_counter  # 更新当前问题ID
                                    
                                    # 生成TTS音频
                                    next_audio_path = os.path.join(TTS_AUDIO_DIR, f"question_{question_counter}.wav")
                                    success = await tts_processor.generate_tts(next_question, next_audio_path)
                                    
                                    if success:
                                        logger.info(f"已生成问题音频: {next_audio_path}")
                                        await tts_processor.send_audio(websocket, next_audio_path)
                                        logger.info(f"已发送下一个问题: {next_question}")
                                        
                                        awaiting_response = True
                                        question_counter += 1
                                        # 清空音频缓冲区，准备接收下一个回答
                                        audio_buffer = np.array([], dtype=np.float32)
                                    else:
                                        logger.error("TTS生成失败，使用预定义的更多细节音频")
                                        await tts_processor.send_audio(websocket, MORE_DETAILS_FILE)
                                else:
                                    logger.error("问题生成失败，使用预定义的更多细节音频")
                                    await tts_processor.send_audio(websocket, MORE_DETAILS_FILE)
                            else:
                                logger.info("已达到最大问题数量，结束面试")
                                await websocket.send("All questions have been asked. Thank you for the interview!")
                                # 播放结束音频
                                await tts_processor.send_audio(websocket, BYE_FILE)
                                break
                            
                            continue
                    
                    # 如果缓冲区有足够的音频数据，进行最终转录
                    if len(audio_buffer) >= MIN_AUDIO_BUFFER_SIZE and awaiting_response:
                        # 保存最终的音频缓冲区用于调试
                        asr_processor.save_debug_audio(audio_buffer, prefix="final_transcribe_buffer")
                        
                        # 进行最终转录
                        transcription, _ = asr_processor.transcribe(audio_buffer, beam_size=5)
                        
                        if transcription and len(transcription) > 0:
                            logger.info(f"最终转录结果: {transcription}")
                            await websocket.send(f"TRANSCRIPTION: {transcription}")
                            
                            # 分析用户回答状态
                            response_status, explanation = await tts_processor.analyze_user_response(
                                transcription, 
                                current_question, 
                                chat_with_ollama
                            )
                            
                            logger.info(f"用户回答状态: {response_status} - {explanation}")
                            
                            # 根据回答状态采取不同行动
                            if response_status == 1:  # 用户没有回答问题
                                logger.info("用户没有回答问题，重新提问")
                                await websocket.send("STATUS: 请回答当前问题")
                                # 可以选择重新播放问题
                                
                            elif response_status == 2:  # 用户正在思考，回答未结束
                                logger.info("用户回答未结束，发送更多细节提示")
                                await websocket.send("STATUS: 请继续您的回答")
                                
                                # 如果还没有发送过更多细节提示，则发送
                                if not more_details_sent:
                                    logger.info("发送更多细节提示音频")
                                    await tts_processor.send_audio(websocket, MORE_DETAILS_FILE)
                                    more_details_sent = True
                                    # 重置连续静音计数
                                    consecutive_silence_count = 0
                                
                            elif response_status == 3:  # 用户已完成回答
                                logger.info("用户已完成回答，准备下一个问题")
                                # 将用户回答添加到对话历史
                                conversation_history.append({"role": "user", "content": transcription})
                                
                                # 重置连续静音计数和更多细节标志
                                consecutive_silence_count = 0
                                more_details_sent = False
                                
                                # 根据用户回答生成下一个问题
                                if question_counter <= MAX_QUESTIONS:
                                    logger.info("根据用户回答生成下一个问题...")
                                    
                                    # 生成下一个问题
                                    next_question = chat_with_ollama(conversation_history)
                                    
                                    if next_question:
                                        logger.info(f"生成的下一个问题: {next_question}")
                                        
                                        # 添加到对话历史
                                        conversation_history.append({"role": "assistant", "content": next_question})
                                        current_question = next_question
                                        current_question_id = question_counter  # 更新当前问题ID
                                        
                                        # 生成TTS音频
                                        next_audio_path = os.path.join(TTS_AUDIO_DIR, f"question_{question_counter}.wav")
                                        success = await tts_processor.generate_tts(next_question, next_audio_path)
                                        
                                        if success:
                                            logger.info(f"已生成问题音频: {next_audio_path}")
                                            await tts_processor.send_audio(websocket, next_audio_path)
                                            logger.info(f"已发送下一个问题: {next_question}")
                                            
                                            awaiting_response = True
                                            question_counter += 1
                                            # 清空音频缓冲区，准备接收下一个回答
                                            audio_buffer = np.array([], dtype=np.float32)
                                        else:
                                            logger.error("TTS生成失败，使用预定义的更多细节音频")
                                            await tts_processor.send_audio(websocket, MORE_DETAILS_FILE)
                                    else:
                                        logger.error("问题生成失败，使用预定义的更多细节音频")
                                        await tts_processor.send_audio(websocket, MORE_DETAILS_FILE)
                                else:
                                    logger.info("已达到最大问题数量，结束面试")
                                    await websocket.send("All questions have been asked. Thank you for the interview!")
                                    # 播放结束音频
                                    await tts_processor.send_audio(websocket, BYE_FILE)
                                    break
                                
                                # 重置超时计时器
                                start_time = time.time()
                                silence_counter = 0
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

                        # 检测是否为有效音频（非静音）
                        current_is_silence = np.mean(np.abs(audio_chunk)) < SILENCE_THRESHOLD
                        current_time = time.time()
                        
                        # 静音状态转换逻辑
                        if current_is_silence and not is_silence:
                            # 从有声音变为静音
                            is_silence = True
                            silence_start_time = current_time
                            logger.debug("检测到静音开始")
                        elif not current_is_silence:
                            # 有声音，重置静音状态
                            is_silence = False
                            silence_start_time = None
                            logger.debug("检测到声音，重置静音状态")
                        
                        # 如果持续静音超过阈值且有足够的转录内容，分析用户回答状态
                        if (is_silence and silence_start_time and 
                                current_time - silence_start_time >= SILENCE_ANALYSIS_THRESHOLD and
                                current_time - last_silence_analysis_time >= SILENCE_ANALYSIS_THRESHOLD and
                                len(audio_buffer) >= MIN_AUDIO_BUFFER_SIZE and awaiting_response):
                            
                            # 更新上次分析时间
                            last_silence_analysis_time = current_time
                            
                            # 进行转录
                            transcription, _ = asr_processor.transcribe(audio_buffer, beam_size=5)
                            
                            if transcription and len(transcription) > 0:
                                logger.info(f"静音分析转录: {transcription}")
                                await websocket.send(f"TRANSCRIPTION: {transcription}")
                                
                                # 分析用户回答状态
                                response_status, explanation = await tts_processor.analyze_user_response(
                                    transcription, 
                                    current_question, 
                                    chat_with_ollama
                                )
                                
                                logger.info(f"静音期间用户回答状态: {response_status} - {explanation}")
                                
                                # 根据回答状态采取不同行动
                                if response_status == 1:  # 用户没有回答问题
                                    # 如果用户没有回答问题，但已经有一些内容，可能是在组织语言
                                    if len(transcription) > 10:
                                        await websocket.send("STATUS: 请继续回答问题")
                                    else:
                                        await websocket.send("STATUS: 请回答当前问题")
                                    
                                elif response_status == 2:  # 用户正在思考，回答未结束
                                    # 如果还没有发送过更多细节提示，则发送
                                    if not more_details_sent:
                                        logger.info("发送更多细节提示音频")
                                        await tts_processor.send_audio(websocket, MORE_DETAILS_FILE)
                                        more_details_sent = True
                                        # 重置连续静音计数
                                        consecutive_silence_count = 0
                                    
                                elif response_status == 3:  # 用户已完成回答
                                    logger.info("用户已完成回答，准备下一个问题")
                                    # 将用户回答添加到对话历史
                                    conversation_history.append({"role": "user", "content": transcription})
                                    
                                    # 重置连续静音计数和更多细节标志
                                    consecutive_silence_count = 0
                                    more_details_sent = False
                                    
                                    # 根据用户回答生成下一个问题
                                    if question_counter <= MAX_QUESTIONS:
                                        logger.info("根据用户回答生成下一个问题...")
                                        
                                        # 生成下一个问题
                                        next_question = chat_with_ollama(conversation_history)
                                        
                                        if next_question:
                                            logger.info(f"生成的下一个问题: {next_question}")
                                            
                                            # 添加到对话历史
                                            conversation_history.append({"role": "assistant", "content": next_question})
                                            current_question = next_question
                                            current_question_id = question_counter  # 更新当前问题ID
                                            
                                            # 生成TTS音频
                                            next_audio_path = os.path.join(TTS_AUDIO_DIR, f"question_{question_counter}.wav")
                                            success = await tts_processor.generate_tts(next_question, next_audio_path)
                                            
                                            if success:
                                                logger.info(f"已生成问题音频: {next_audio_path}")
                                                await tts_processor.send_audio(websocket, next_audio_path)
                                                logger.info(f"已发送下一个问题: {next_question}")
                                                
                                                awaiting_response = True
                                                question_counter += 1
                                                # 清空音频缓冲区，准备接收下一个回答
                                                audio_buffer = np.array([], dtype=np.float32)
                                            else:
                                                logger.error("TTS生成失败，使用预定义的更多细节音频")
                                                await tts_processor.send_audio(websocket, MORE_DETAILS_FILE)
                                        else:
                                            logger.error("问题生成失败，使用预定义的更多细节音频")
                                            await tts_processor.send_audio(websocket, MORE_DETAILS_FILE)
                                    else:
                                        logger.info("已达到最大问题数量，结束面试")
                                        await websocket.send("All questions have been asked. Thank you for the interview!")
                                        # 播放结束音频
                                        await tts_processor.send_audio(websocket, BYE_FILE)
                                        break

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

                    # 检测是否为有效音频（非静音）
                    current_is_silence = np.mean(np.abs(audio_chunk)) < SILENCE_THRESHOLD
                    current_time = time.time()
                    
                    # 静音状态转换逻辑
                    if current_is_silence and not is_silence:
                        # 从有声音变为静音
                        is_silence = True
                        silence_start_time = current_time
                        logger.debug("检测到静音开始")
                    elif not current_is_silence:
                        # 有声音，重置静音状态
                        is_silence = False
                        silence_start_time = None
                        logger.debug("检测到声音，重置静音状态")
                    
                    # 如果持续静音超过阈值且有足够的转录内容，分析用户回答状态
                    if (is_silence and silence_start_time and 
                            current_time - silence_start_time >= SILENCE_ANALYSIS_THRESHOLD and
                            current_time - last_silence_analysis_time >= SILENCE_ANALYSIS_THRESHOLD and
                            len(audio_buffer) >= MIN_AUDIO_BUFFER_SIZE and awaiting_response):
                        
                        # 更新上次分析时间
                        last_silence_analysis_time = current_time
                        
                        # 进行转录
                        transcription, _ = asr_processor.transcribe(audio_buffer, beam_size=5)
                        
                        if transcription and len(transcription) > 0:
                            logger.info(f"静音分析转录: {transcription}")
                            await websocket.send(f"TRANSCRIPTION: {transcription}")
                            
                            # 分析用户回答状态
                            response_status, explanation = await tts_processor.analyze_user_response(
                                transcription, 
                                current_question, 
                                chat_with_ollama
                            )
                            
                            logger.info(f"静音期间用户回答状态: {response_status} - {explanation}")
                            
                            # 根据回答状态采取不同行动
                            if response_status == 1:  # 用户没有回答问题
                                # 如果用户没有回答问题，但已经有一些内容，可能是在组织语言
                                if len(transcription) > 10:
                                    await websocket.send("STATUS: 请继续回答问题")
                                else:
                                    await websocket.send("STATUS: 请回答当前问题")
                                
                            elif response_status == 2:  # 用户正在思考，回答未结束
                                # 如果还没有发送过更多细节提示，则发送
                                if not more_details_sent:
                                    logger.info("发送更多细节提示音频")
                                    await tts_processor.send_audio(websocket, MORE_DETAILS_FILE)
                                    more_details_sent = True
                                    # 重置连续静音计数
                                    consecutive_silence_count = 0
                                
                            elif response_status == 3:  # 用户已完成回答
                                logger.info("用户已完成回答，准备下一个问题")
                                # 将用户回答添加到对话历史
                                conversation_history.append({"role": "user", "content": transcription})
                                
                                # 重置连续静音计数和更多细节标志
                                consecutive_silence_count = 0
                                more_details_sent = False
                                
                                # 根据用户回答生成下一个问题
                                if question_counter <= MAX_QUESTIONS:
                                    logger.info("根据用户回答生成下一个问题...")
                                    
                                    # 生成下一个问题
                                    next_question = chat_with_ollama(conversation_history)
                                    
                                    if next_question:
                                        logger.info(f"生成的下一个问题: {next_question}")
                                        
                                        # 添加到对话历史
                                        conversation_history.append({"role": "assistant", "content": next_question})
                                        current_question = next_question
                                        current_question_id = question_counter  # 更新当前问题ID
                                        
                                        # 生成TTS音频
                                        next_audio_path = os.path.join(TTS_AUDIO_DIR, f"question_{question_counter}.wav")
                                        success = await tts_processor.generate_tts(next_question, next_audio_path)
                                        
                                        if success:
                                            logger.info(f"已生成问题音频: {next_audio_path}")
                                            await tts_processor.send_audio(websocket, next_audio_path)
                                            logger.info(f"已发送下一个问题: {next_question}")
                                            
                                            awaiting_response = True
                                            question_counter += 1
                                            # 清空音频缓冲区，准备接收下一个回答
                                            audio_buffer = np.array([], dtype=np.float32)
                                        else:
                                            logger.error("TTS生成失败，使用预定义的更多细节音频")
                                            await tts_processor.send_audio(websocket, MORE_DETAILS_FILE)
                                    else:
                                        logger.error("问题生成失败，使用预定义的更多细节音频")
                                        await tts_processor.send_audio(websocket, MORE_DETAILS_FILE)
                                else:
                                    logger.info("已达到最大问题数量，结束面试")
                                    await websocket.send("All questions have been asked. Thank you for the interview!")
                                    # 播放结束音频
                                    await tts_processor.send_audio(websocket, BYE_FILE)
                                    break
                    
                    denoised_chunk = audio_chunk

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