import numpy as np
import logging
import time
import os
import soundfile as sf

# 获取logger
logger = logging.getLogger(__name__)

class AudioSender:
    """处理音频数据的发送逻辑"""
    def __init__(self, sample_rate=16000, debug_dir="client_debug_audio", 
                 vad_threshold=0.01, min_speech_duration=0.3):
        self.websocket = None
        self.debug_dir = debug_dir
        self.send_count = 0
        self.sample_rate = sample_rate
        
        # VAD相关参数
        self.vad_threshold = vad_threshold  # 语音活动检测阈值
        self.min_speech_frames = int(min_speech_duration * sample_rate)  # 最小语音帧数
        
        # 确保调试目录存在
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def set_websocket(self, websocket):
        """设置WebSocket连接"""
        self.websocket = websocket
        logger.info("已设置WebSocket连接到音频发送器")
    
    def save_debug_audio(self, audio_data, prefix="debug"):
        """保存调试音频文件，使用递增编号"""
        # 使用全局计数器或类内部计数器
        self.send_count += 1 if prefix.startswith("send") else 0
        
        try:
            # 使用递增的编号命名文件
            debug_file = os.path.join(self.debug_dir, f"{prefix}_{self.send_count:04d}_{time.time():.2f}.wav")
            
            # 确保音频数据是float32类型
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            # 保存音频文件
            sf.write(debug_file, audio_data, self.sample_rate)
            logger.debug(f"已保存客户端调试音频到: {debug_file}, 长度: {len(audio_data)/self.sample_rate:.2f}秒")
            
            # 如果音频文件太多，删除旧文件
            self.cleanup_debug_files(max_files=100)
            
            return debug_file
        except Exception as e:
            logger.warning(f"无法保存客户端调试音频: {e}")
            return None
    
    def cleanup_debug_files(self, max_files=100):
        """清理旧的调试文件，保留最新的max_files个文件"""
        try:
            files = [os.path.join(self.debug_dir, f) for f in os.listdir(self.debug_dir) 
                    if f.endswith('.wav')]
            
            if len(files) > max_files:
                # 按修改时间排序
                files.sort(key=os.path.getmtime)
                
                # 删除最旧的文件
                for file_to_delete in files[:-max_files]:
                    os.remove(file_to_delete)
                    logger.debug(f"已删除旧的客户端调试音频: {file_to_delete}")
        except Exception as e:
            logger.warning(f"清理客户端调试文件时出错: {e}")
    
    def detect_speech_segments(self, audio_data, frame_size=160):
        """
        检测音频中的语音段，返回语音段的列表
        frame_size: 每个分析帧的大小，默认为10ms (160个样本，在16kHz下)
        返回: 语音段列表，每个元素为(start_idx, end_idx)
        """
        if len(audio_data) == 0:
            return []
            
        # 计算每帧的能量
        num_frames = len(audio_data) // frame_size
        energies = np.zeros(num_frames)
        
        for i in range(num_frames):
            frame = audio_data[i*frame_size:(i+1)*frame_size]
            energies[i] = np.mean(frame**2)
        
        # 检测语音段
        is_speech = energies > self.vad_threshold
        
        # 找出语音段的起始和结束位置
        speech_segments = []
        in_speech = False
        start_idx = 0
        
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                # 语音开始
                in_speech = True
                start_idx = i * frame_size
            elif not speech and in_speech:
                # 语音结束
                in_speech = False
                end_idx = i * frame_size
                # 只保留足够长的语音段
                if end_idx - start_idx >= self.min_speech_frames:
                    speech_segments.append((start_idx, end_idx))
        
        # 处理最后一个语音段
        if in_speech:
            end_idx = len(audio_data)
            if end_idx - start_idx >= self.min_speech_frames:
                speech_segments.append((start_idx, end_idx))
        
        # 合并相近的语音段
        if len(speech_segments) > 1:
            merged_segments = [speech_segments[0]]
            for segment in speech_segments[1:]:
                prev_end = merged_segments[-1][1]
                curr_start = segment[0]
                # 如果两个语音段间隔小于0.5秒，则合并
                if curr_start - prev_end < 0.5 * self.sample_rate:
                    merged_segments[-1] = (merged_segments[-1][0], segment[1])
                else:
                    merged_segments.append(segment)
            speech_segments = merged_segments
            
        return speech_segments
    
    def extract_speech_audio(self, audio_data):
        """
        提取音频中的语音部分，过滤掉静音
        返回: 只包含语音的音频数据
        """
        # 检测语音段
        speech_segments = self.detect_speech_segments(audio_data)
        
        if not speech_segments:
            logger.info("未检测到语音内容，返回原始音频")
            return audio_data
        
        # 提取语音段
        speech_audio = []
        for start, end in speech_segments:
            speech_audio.append(audio_data[start:end])
        
        # 合并所有语音段
        if speech_audio:
            combined_speech = np.concatenate(speech_audio)
            logger.info(f"提取了 {len(speech_segments)} 个语音段，总长度: {len(combined_speech)/self.sample_rate:.2f}秒")
            
            # 保存提取的语音段
            self.save_debug_audio(combined_speech, prefix="speech_extracted")
            
            return combined_speech
        else:
            logger.debug("未提取到有效语音段，返回原始音频")
            return audio_data
    
    async def send_audio_data(self, audio_data):
        """发送单个音频数据块"""
        if len(audio_data) == 0:
            logger.debug("尝试发送空音频缓冲区，已跳过")
            return False
        
        if not self.websocket:
            logger.debug("WebSocket连接未设置，无法发送音频数据")
            return False
        
        # 保存原始音频数据
        self.save_debug_audio(audio_data, prefix=f"original_{self.send_count}")
        
        # 提取语音内容
        speech_audio = self.extract_speech_audio(audio_data)
        
        # 如果提取的语音内容太短，可能不包含有效语音，跳过发送
        if len(speech_audio) < 0.5 * self.sample_rate:  # 小于0.5秒
            logger.debug(f"提取的语音内容太短 ({len(speech_audio)/self.sample_rate:.2f}秒)，跳过发送")
            return False
        
        logger.debug(f"发送音频数据到服务端 #{self.send_count}: 形状={speech_audio.shape}, 类型={speech_audio.dtype}, 最大值={np.max(np.abs(speech_audio))}, 时长={len(speech_audio)/self.sample_rate:.2f}秒")

        # 保存发送前的音频数据
        self.save_debug_audio(speech_audio, prefix=f"send_{self.send_count}")

        # 确保音频数据是float32类型
        if speech_audio.dtype != np.float32:
            speech_audio = speech_audio.astype(np.float32)
            
        # 确保音频数据范围在[-1, 1]之间
        if np.max(np.abs(speech_audio)) > 1.0:
            speech_audio = speech_audio / np.max(np.abs(speech_audio))

        # 始终增加音量，确保音频信号足够强
        # 即使音量很小也增加增益
        max_val = np.max(np.abs(speech_audio))
        if max_val < 0.5:  # 提高阈值
            gain = 0.5 / max_val if max_val > 0 else 5.0
            speech_audio = speech_audio * min(gain, 10.0)  # 增加最大增益到10倍
            logger.debug(f"增加音频音量，增益={min(gain, 10.0)}")
            
            # 保存增益后的音频数据
            self.save_debug_audio(speech_audio, prefix=f"send_gain_{self.send_count}")
        
        try:
            # 将二进制数据转换为latin1编码的字符串，确保可以通过websocket传输
            audio_bytes = speech_audio.tobytes()
            encoded_audio = audio_bytes.decode('latin1')
            
            # 保存最终发送到服务端的音频数据（在编码前）
            self.save_debug_audio(speech_audio, prefix=f"final_send_{self.send_count}")

            # 发送带前缀的音频数据
            await self.websocket.send(f"AUDIO:{encoded_audio}")
            
            # 添加调试信息，确认数据已发送
            logger.debug(f"已发送音频数据 #{self.send_count}，大小: {len(speech_audio)} 样本，时长: {len(speech_audio)/self.sample_rate:.2f}秒")
            return True
        except Exception as e:
            logger.error(f"发送音频数据时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # 重新抛出异常以便上层函数可以处理
    
    async def send_control_message(self, message):
        """发送控制消息到服务器"""
        if not self.websocket:
            logger.warning(f"WebSocket连接未设置，无法发送控制消息: {message}")
            return False
        
        try:
            await self.websocket.send(message)
            logger.debug(f"已发送控制消息: {message}")
            return True
        except Exception as e:
            logger.error(f"发送控制消息时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def process_audio_queue(self, audio_queue, running_flag):
        """处理音频队列并发送数据"""
        import asyncio
        from queue import Full, Empty

        logger.debug("开始处理音频队列 process_audio_queue")
        
        buffer = np.array([], dtype=np.float32)  # 创建本地缓冲区
        last_send_time = time.time()
        last_log_time = time.time()
        buffer_count = 0  # 跟踪添加到缓冲区的块数
        
        # 创建一个缓存，用于存储最近的音频数据
        recent_audio_buffer = np.array([], dtype=np.float32)
        max_buffer_duration = 10  # 最大缓存10秒的音频

        # 添加队列获取失败计数
        queue_get_failures = 0
        last_queue_size = 0

        # 保存一个调试文件，记录处理队列开始
        debug_file = os.path.join(self.debug_dir, f"process_queue_start_{time.time():.2f}.txt")
        with open(debug_file, 'w') as f:
            f.write(f"开始处理音频队列，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")        
        
        while running_flag and self.websocket:
            try:
                # 定期记录缓冲区状态
                current_time = time.time()
                if current_time - last_log_time > 5.0:
                    queue_size = audio_queue.qsize()
                    logger.debug(f"音频队列状态: 大小={queue_size}/{audio_queue.maxsize}, 缓冲区={len(buffer)/self.sample_rate:.2f}秒")

                    # 只有当队列不为空时才记录详细状态
                    if queue_size == last_queue_size and queue_size > 0:
                        logger.warning(f"队列大小在过去5秒内没有变化: {queue_size}")
                    else:
                        # 队列为空时只记录debug级别的日志
                        logger.debug(f"队列为空，等待输入...")

                    last_queue_size = queue_size
                    last_log_time = current_time


                # 尝试直接从队列获取数据（非阻塞方式）
                try:
                    # 使用非阻塞方式获取数据
                    audio_data = audio_queue.get_nowait()
                    queue_get_failures = 0 # 重置失败计数

                    # 检查是否为静音标记
                    if len(audio_data) == 1 and np.all(audio_data == 0):
                        logger.info("收到静音标记，发送当前缓冲区并通知服务器用户停止说话")
                        
                        # 如果缓冲区有数据，先发送
                        if len(buffer) > 0:
                            await self.send_audio_data(buffer)
                            buffer = np.array([], dtype=np.float32)
                            buffer_count = 0
                        
                        # 发送静音通知
                        await self.send_control_message("SILENCE_DETECTED")
                    else:
                        # 将新的音频数据添加到本地缓冲区
                        buffer = np.concatenate((buffer, audio_data))
                        buffer_count += 1
                        logger.debug(f"添加音频到缓冲区: 当前大小={len(buffer)/self.sample_rate:.2f}秒, 块数={buffer_count}")
                        
                        # 当缓冲区达到一定大小时发送
                        if len(buffer) >= 2 * self.sample_rate:  # 2秒的音频
                            logger.debug(f"缓冲区达到发送阈值: {len(buffer)/self.sample_rate:.2f}秒")
                            await self.send_audio_data(buffer)
                            buffer = np.array([], dtype=np.float32)
                            buffer_count = 0
                            last_send_time = current_time
                
                    # 标记任务完成
                    audio_queue.task_done()
                    
                except Empty:
                    # 队列为空，等待一小段时间
                    queue_get_failures += 1
                    
                    # 如果缓冲区有数据且已经累积了足够长的时间，即使队列为空也发送
                    if len(buffer) > 0 and (current_time - last_send_time >= 3.0):
                        logger.debug(f"队列为空但缓冲区有数据，发送现有缓冲区: {len(buffer)/self.sample_rate:.2f}秒")
                        await self.send_audio_data(buffer)
                        buffer = np.array([], dtype=np.float32)
                        buffer_count = 0
                        last_send_time = current_time
                    
                    # 避免CPU占用过高
                    await asyncio.sleep(0.1)
                    
                    # 只在第一次失败和每10次失败后记录日志
                    if queue_get_failures == 1 or queue_get_failures % 10 == 0:
                        logger.debug(f"队列为空，等待数据... (尝试次数: {queue_get_failures})")
                    
                    continue
            
            except Exception as e:
                logger.error(f"处理音频队列时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # 不退出循环，尝试继续处理
                await asyncio.sleep(0.5)
                
        # 如果退出循环时缓冲区还有数据，发送剩余数据
        if len(buffer) > 0:
            logger.debug(f"处理结束，发送剩余缓冲区: {len(buffer)/self.sample_rate:.2f}秒")
            try:
                await self.send_audio_data(buffer)
            except Exception as e:
                logger.error(f"发送剩余缓冲区时出错: {e}")
        
        logger.info("音频队列处理完成")               