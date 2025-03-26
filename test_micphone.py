import sounddevice as sd
import numpy as np
import logging
import time
import sys
from faster_whisper import WhisperModel
import queue
import threading

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 音频常量
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000
CHANNELS = 1
DURATION = 30  # 测试持续时间（秒），设为None则一直运行

# 初始化Whisper模型
MODEL_PATH = "large"  # 可以根据需要改为"tiny"、"base"等更小的模型
COMPUTE_TYPE = "float16"  # 如果没有GPU，可以改为"int8"

class MicrophoneTest:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.running = True
        self.stream = None
        self.model = None
        self.audio_buffer = np.array([], dtype=np.float32)
        
    def load_model(self):
        """加载Whisper模型"""
        try:
            logger.info("正在加载Whisper模型...")
            # 尝试使用GPU
            try:
                self.model = WhisperModel(MODEL_PATH, compute_type=COMPUTE_TYPE, device="cuda")
                logger.info("已加载GPU模型")
            except Exception as e:
                logger.warning(f"无法加载GPU模型: {e}，将使用CPU模型")
                self.model = WhisperModel(MODEL_PATH, compute_type=COMPUTE_TYPE, device="cpu")
                logger.info("已加载CPU模型")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
            
    def audio_callback(self, indata, frames, time, status):
        """麦克风输入的回调函数"""
        if status:
            logger.warning(f"音频回调状态: {status}")
        if self.running:
            audio_data = indata[:, 0].astype(np.float32)
            self.audio_queue.put(audio_data)
            
    def process_audio(self):
        """处理音频队列中的数据并进行转录"""
        while self.running:
            try:
                # 从队列获取音频数据
                audio_data = self.audio_queue.get(timeout=1)
                
                # 添加到缓冲区
                self.audio_buffer = np.concatenate((self.audio_buffer, audio_data))
                
                # 当缓冲区足够大时进行转录
                if len(self.audio_buffer) > SAMPLE_RATE * 2:  # 至少2秒的音频
                    self.transcribe_audio()
                    
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"处理音频时出错: {e}")
                
    def transcribe_audio(self):
        """转录音频缓冲区中的内容"""
        if self.model is None or len(self.audio_buffer) == 0:
            return
            
        try:
            # 进行转录
            segments, _ = self.model.transcribe(self.audio_buffer, beam_size=5)
            transcription = " ".join(segment.text for segment in segments)
            
            if transcription.strip():
                logger.info(f"转录结果: {transcription}")
                
            # 保留最后1秒的音频，以便连续转录
            self.audio_buffer = self.audio_buffer[-SAMPLE_RATE:]
            
        except Exception as e:
            logger.error(f"转录错误: {e}")
            
    def start(self):
        """启动麦克风测试"""
        if not self.load_model():
            logger.error("无法加载模型，测试终止")
            return
            
        logger.info("开始麦克风测试，请对着麦克风说话...")
        
        # 创建并启动音频处理线程
        processing_thread = threading.Thread(target=self.process_audio)
        processing_thread.daemon = True
        processing_thread.start()
        
        # 启动音频流
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype='float32',
            callback=self.audio_callback
        )
        
        with self.stream:
            try:
                # 如果设置了持续时间，则运行指定时间
                if DURATION:
                    logger.info(f"测试将持续 {DURATION} 秒...")
                    time.sleep(DURATION)
                else:
                    logger.info("测试运行中，按Ctrl+C停止...")
                    while True:
                        time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("用户中断测试")
            finally:
                self.stop()
                
    def stop(self):
        """停止测试"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        logger.info("麦克风测试已停止")

if __name__ == "__main__":
    test = MicrophoneTest()
    test.start()