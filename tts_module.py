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
        
        # 确保音频目录存在
        if not os.path.exists(self.tts_audio_dir):
            os.makedirs(self.tts_audio_dir)
            
        logger.info(f"TTS处理器初始化完成，API地址: {tts_api_url}")
    
    async def generate_audio_async(self, text, filename, tts_queue=None, task_id=None):
        """异步生成TTS音频并可选择将结果放入队列"""
        start_time = time.time()
        task_info = f"[任务 {task_id}] " if task_id else ""
        logger.info(f"{task_info}开始生成TTS音频: {filename}")
        
        try:
            response = requests.post(
                self.tts_api_url,
                json={
                    "ref_audio": self.ref_audio,
                    "ref_text": self.ref_text,
                    "gen_text": text
                }
            )
            response.raise_for_status()
            audio_content = response.content
            
            audio_file_path = os.path.join(self.tts_audio_dir, filename)
            with open(audio_file_path, "wb") as f:
                f.write(audio_content)
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"{task_info}TTS音频生成完成: {audio_file_path} (用时: {duration:.2f}秒)")
            
            # 如果提供了队列，将生成的音频文件路径和问题文本放入队列
            if tts_queue:
                await tts_queue.put((audio_file_path, text))
                
            return audio_file_path
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
            with open(audio_file_path, "rb") as f:
                audio_content = f.read()
            await websocket.send(audio_content)
            logger.info(f"发送音频文件: {audio_file_path}")
            return True
        except Exception as e:
            logger.error(f"发送音频文件错误: {e}")
            await websocket.send(f"Error: Could not send audio file {audio_file_path}")
            return False

    # 修改音频发送部分，添加AUDIO:前缀
    async def send_audio_with_prefix(self, websocket, audio_file_path):
        """发送带AUDIO:前缀的音频数据"""
        with open(audio_file_path, 'rb') as f:
            audio_data = f.read()
            # 添加base64编码
            import base64
            encoded_audio = base64.b64encode(audio_data).decode('utf-8')
            await websocket.send(f"AUDIO:{encoded_audio}")   
            return True         
    
    async def generate_all_questions_async(self, chat_function, conversation_history, start_counter, max_questions, tts_queue):
        """完全异步生成所有问题音频"""
        question_counter = start_counter
        tasks = []
        task_info = {}  # 存储任务信息
        
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