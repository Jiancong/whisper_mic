import os
import time
import logging
import asyncio
import requests

# 配置日志
logger = logging.getLogger(__name__)

class TTSProcessor:
    def __init__(self, tts_api_url, tts_audio_dir, ref_audio="audio_samples/elon-musk-short-sample.wav", 
                 ref_text="So I want to show the people the most importantly, that this is possible. That's the future we could have."):
        """初始化TTS处理器"""
        self.tts_api_url = tts_api_url
        self.tts_audio_dir = tts_audio_dir
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.nfe_step = 16 # default is 32
        
        # 确保音频目录存在
        if not os.path.exists(self.tts_audio_dir):
            os.makedirs(self.tts_audio_dir)
            
        logger.info(f"TTS处理器初始化完成，API地址: {tts_api_url}")

    def check_questions_ready(self, questions_dir, num_questions):
        """检查指定数量的问题是否已经生成完毕"""
        try:
            # 检查目录是否存在
            if not os.path.exists(questions_dir):
                logger.warning(f"问题目录不存在: {questions_dir}")
                return False
                
            # 检查每个问题文件是否存在
            for i in range(2, num_questions + 1):
                question_file = os.path.join(questions_dir, f"question_{i}.wav")
                if not os.path.exists(question_file):
                    logger.info(f"问题文件不存在: {question_file}")
                    return False
            
            logger.info(f"所有 {num_questions} 个问题已准备就绪")
            return True
        except Exception as e:
            logger.error(f"检查问题状态时出错: {e}")
            return False
    
    def count_existing_questions(self, questions_dir):
        """计算已存在的问题数量"""
        try:
            if not os.path.exists(questions_dir):
                return 0
                
            # 计算问题_*.wav文件的数量
            question_files = [f for f in os.listdir(questions_dir) 
                             if f.startswith("question_") and f.endswith(".wav")]
            return len(question_files)
        except Exception as e:
            logger.error(f"计算问题数量时出错: {e}")
            return 0        
    
    async def generate_audio_async(self, text, filename, tts_queue=None, task_id=None):
        """异步生成TTS音频并可选择将结果放入队列"""
        start_time = time.time()
        task_info = f"[任务 {task_id}] " if task_id else ""
        logger.info(f"{task_info}开始生成TTS音频: {filename}")
        
        try:
            # 确保输出目录存在
            audio_file_path = os.path.join(self.tts_audio_dir, filename)
            output_dir = os.path.dirname(audio_file_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)          

            # 准备请求数据
            data = {
                "ref_audio": self.ref_audio,
                "ref_text": self.ref_text,
                "gen_text": text,
                "nfe_step": self.nfe_step,
                "output_file": audio_file_path  # 传递完整的输出路径
            }                  

            response = requests.post(
                self.tts_api_url,
                json=data
            )
            response.raise_for_status()
            audio_content = response.content
            
            audio_file_path = os.path.join(self.tts_audio_dir, filename)
            with open(audio_file_path, "wb") as f:
                f.write(audio_content)
            
            # 验证文件是否成功创建
            if os.path.exists(audio_file_path):
                file_size = os.path.getsize(audio_file_path)
                if file_size > 0:
                    end_time = time.time()
                    duration = end_time - start_time
                    logger.info(f"{task_info}TTS音频生成成功: {audio_file_path} (用时: {duration:.2f}秒), 文件大小: {file_size/1024:.2f}KB")
                    
                    # 如果提供了队列，将生成的音频文件路径和问题文本放入队列
                    if tts_queue:
                        await tts_queue.put((audio_file_path, text))
                        
                    return audio_file_path
                else:
                    logger.error(f"{task_info}TTS音频文件生成成功但大小为0: {audio_file_path}")
                    return None
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"{task_info}TTS生成错误 (用时: {duration:.2f}秒): {e}")
            return None
    
    async def generate_audio(self, text, filename):
        """同步风格的TTS生成函数（为了兼容性）"""
        return await self.generate_audio_async(text, filename)
    
    async def send_audio(self, websocket, audio_file_path):
        """发送TTS音频到客户端"""
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_file_path):
                logger.error(f"发送音频文件错误: [Errno 2] No such file or directory: '{audio_file_path}'")
                
                # 尝试在tts_audio目录中查找文件
                filename = os.path.basename(audio_file_path)
                alternative_path = os.path.join(self.tts_audio_dir, filename)
                
                if os.path.exists(alternative_path):
                    logger.info(f"找到替代文件路径: {alternative_path}")
                    audio_file_path = alternative_path
                else:
                    logger.error(f"替代文件也不存在: {alternative_path}")
                    await websocket.send(f"Error: Could not send audio file {audio_file_path}")
                    return False

            # 检查文件大小
            file_size = os.path.getsize(audio_file_path)
            if file_size == 0:
                logger.error(f"音频文件大小为0: {audio_file_path}")
                await websocket.send(f"Error: Audio file is empty {audio_file_path}")
                return False
                
            logger.info(f"准备发送音频文件: {audio_file_path}, 大小: {file_size/1024:.2f}KB")            

            with open(audio_file_path, "rb") as f:
                audio_content = f.read()

            await websocket.send(audio_content)
            logger.info(f"发送音频文件: {audio_file_path}")
            return True
        except Exception as e:
            logger.error(f"发送音频文件错误: {e}")
            await websocket.send(f"Error: Could not send audio file {audio_file_path}")
            return False
    
    async def generate_all_questions_async(self, chat_function, conversation_history, start_counter, max_questions, tts_queue):
        """完全异步生成所有问题音频"""
        question_counter = start_counter
        tasks = []
        task_info = {}  # 存储任务信息

        history_copy = conversation_history.copy()
        
        while question_counter <= max_questions:
            try:
                # 记录开始时间
                question_start_time = time.time()
                logger.info(f"开始生成问题 {question_counter} 的文本...")
                
                next_question = chat_function(conversation_history + [
                    {"role": "user", "content": "Generate the next interview question based on the candidate's response."}
                ])
                
                # 记录文本生成完成时间
                question_end_time = time.time()
                question_duration = question_end_time - question_start_time
                
                if next_question:
                    logger.info(f"问题 {question_counter} 文本生成完成 (用时: {question_duration:.2f}秒): {next_question}")
                    
                    filename = f"question_{question_counter}.wav"
                    task_id = f"Q{question_counter}"
                    
                    # 创建异步任务但不等待它完成
                    task = asyncio.create_task(
                        self.generate_audio_async(next_question, filename, tts_queue, task_id)
                    )
                    
                    # 记录任务信息
                    task_info[task_id] = {
                        "question_number": question_counter,
                        "text": next_question,
                        "start_time": time.time(),
                        "status": "pending"
                    }
                    
                    # 添加任务完成回调
                    task.add_done_callback(
                        lambda t, tid=task_id: self._task_completed_callback(t, tid, task_info)
                    )
                    
                    tasks.append(task)
                    logger.info(f"创建问题 {question_counter} 的TTS生成任务 (ID: {task_id})")
                    question_counter += 1
                else:
                    logger.error("Failed to generate next question from API.")
                    break
                    
            except Exception as e:
                logger.error(f"Error generating question {question_counter}: {e}")
                break
        
        # 创建任务监控器
        monitor_task = asyncio.create_task(self._monitor_tts_tasks(tasks, task_info))
        tasks.append(monitor_task)
                
        # 返回创建的任务数量，但不等待它们完成
        return question_counter - 1, tasks
    
    def _task_completed_callback(self, task, task_id, task_info):
        """当任务完成时更新任务信息"""
        if task_id in task_info:
            end_time = time.time()
            duration = end_time - task_info[task_id]["start_time"]
            
            if task.exception():
                task_info[task_id]["status"] = "failed"
                task_info[task_id]["error"] = str(task.exception())
                logger.error(f"[任务 {task_id}] 失败 (用时: {duration:.2f}秒): {task.exception()}")
            else:
                task_info[task_id]["status"] = "completed"
                task_info[task_id]["end_time"] = end_time
                task_info[task_id]["duration"] = duration
                logger.info(f"[任务 {task_id}] 完成 (用时: {duration:.2f}秒)")
    
    async def _monitor_tts_tasks(self, tasks, task_info):
        """监控TTS任务的完成情况"""
        try:
            # 初始状态报告
            logger.info(f"开始监控 {len(tasks)-1} 个TTS生成任务")
            
            # 每5秒检查一次任务状态
            while not all(task.done() for task in tasks[:-1]):  # 排除监控任务本身
                pending_count = sum(1 for task in tasks[:-1] if not task.done())
                completed_count = len(tasks) - 1 - pending_count
                
                # 计算已完成任务的平均耗时
                completed_durations = [
                    info.get("duration", 0) 
                    for info in task_info.values() 
                    if info.get("status") == "completed"
                ]
                
                avg_duration = sum(completed_durations) / len(completed_durations) if completed_durations else 0
                
                logger.info(f"TTS任务状态: {completed_count}/{len(tasks)-1} 完成 (平均耗时: {avg_duration:.2f}秒)")
                
                # 输出每个任务的状态
                for task_id, info in task_info.items():
                    status = info.get("status", "pending")
                    if status == "pending":
                        elapsed = time.time() - info["start_time"]
                        logger.info(f"  - [任务 {task_id}] 问题 {info['question_number']}: 进行中 (已耗时: {elapsed:.2f}秒)")
                    
                await asyncio.sleep(5)
            
            # 最终状态报告
            logger.info("所有TTS任务已完成")
            for task_id, info in task_info.items():
                status = info.get("status", "unknown")
                duration = info.get("duration", 0)
                logger.info(f"  - [任务 {task_id}] 问题 {info['question_number']}: {status} (耗时: {duration:.2f}秒)")
                
        except Exception as e:
            logger.error(f"任务监控器错误: {e}")

# ... existing code ...

    async def analyze_user_response(self, transcription, current_question, chat_function):
        """分析用户回答状态
        返回值:
        1 - 用户没有回答问题
        2 - 用户正在思考，回答未结束
        3 - 用户已完成回答
        4 - 用户不想回答当前问题
        """

        try:
            # 构建分析提示
            prompt = [
                {
                    "role": "system",
                    "content": "你是一个面试助手，负责分析用户的回答是否完整。请根据问题和回答内容，判断用户的回答状态。"
                },
                {
                    "role": "user",
                    "content": f"问题: {current_question}\n\n用户回答: {transcription}\n\n请分析用户回答状态，并返回以下三种状态之一:\n1 - 用户没有回答问题\n2 - 用户正在思考，回答未结束\n3 - 用户已完成回答\n 4 - 用户拒绝回答当前问题 \n\n只需返回数字和简短解释，格式为: '状态数字:解释'"
                }
            ]
            
            # 调用LLM分析
            response = chat_function(prompt)
            
            if response:
                logger.info(f"回答分析结果: {response}")
                
                # 尝试从回答中提取状态数字
                if "1:" in response or "1 -" in response or "1：" in response or "状态1" in response:
                    return 1, "用户没有回答问题"
                elif "2:" in response or "2 -" in response or "2：" in response or "状态2" in response:
                    return 2, "用户正在思考，回答未结束"
                elif "3:" in response or "3 -" in response or "3：" in response or "状态3" in response:
                    return 3, "用户已完成回答"
                elif "4:" in response or "4 -" in response or "4：" in response or "状态4" in response:
                    return 3, "用户拒绝回答该问题"                    
                else:
                    # 如果无法确定状态，默认为回答未结束
                    return 2, "无法确定状态，默认为回答未结束"
            else:
                logger.error("分析用户回答状态失败，LLM返回为空")
                return 2, "分析失败，默认为回答未结束"
                
        except Exception as e:
            logger.error(f"分析用户回答状态错误: {e}")
            return 2, f"分析错误: {str(e)}"


    # 添加一个新方法用于生成单个问题的TTS音频
    async def generate_tts(self, text, output_path):
        """生成单个问题的TTS音频"""
        try:
            logger.info(f"开始生成TTS音频: {output_path}")
            start_time = time.time()
            
            # 准备请求数据
            data = {
                "voice": "elon_musk",  # 使用预设的声音
                "output_file": output_path,
                "ref_audio": self.ref_audio,
                "ref_text": self.ref_text,
                "gen_text": text,
                "nfe_step": self.nfe_step
            }
            
            # 发送请求到TTS服务
            response = requests.post(self.tts_api_url, json=data)
            
            if response.status_code == 200:
                # 将响应内容保存到文件
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                # 验证文件是否成功创建
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    if file_size > 0:
                        end_time = time.time()
                        duration = end_time - start_time
                        logger.info(f"TTS音频生成成功 (用时: {duration:.2f}秒): {output_path}, 文件大小: {file_size/1024:.2f}KB")
                        return True
                    else:
                        logger.error(f"TTS音频文件生成成功但大小为0: {output_path}")
                        return False
                else:
                    logger.error(f"TTS音频文件未成功创建: {output_path}")
                    return False
            else:
                logger.error(f"TTS生成错误 (HTTP {response.status_code}): {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"TTS生成错误: {e}")
            return False            