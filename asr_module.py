import numpy as np
import logging
from faster_whisper import WhisperModel
import librosa
import time  # 添加time模块导入
import os
import soundfile as sf

# 配置日志
logger = logging.getLogger(__name__)

# 添加一个全局计数器用于音频文件命名
debug_counter = 0

class ASRProcessor:
    def __init__(self, model_path="medium", compute_type="float16", device="cuda", sample_rate=16000):
        """初始化ASR处理器"""
        self.sample_rate = sample_rate
        logger.info(f"加载Whisper模型: {model_path}...")
        self.model = WhisperModel(model_path, compute_type=compute_type, device=device)
        logger.info(f"模型加载完成，使用设备: {device}")

        # 创建调试音频目录
        self.debug_dir = "debug_audio"
        os.makedirs(self.debug_dir, exist_ok=True)

    def save_debug_audio(self, audio_data, prefix="debug"):
        """保存调试音频文件，使用递增编号"""
        global debug_counter
        debug_counter += 1
        
        try:
            # 使用递增的编号命名文件
            debug_file = os.path.join(self.debug_dir, f"{prefix}_{debug_counter:04d}_{time.time():.2f}.wav")
            
            # 确保音频数据是float32类型
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            # 保存音频文件
            sf.write(debug_file, audio_data, self.sample_rate)
            logger.info(f"已保存调试音频到: {debug_file}, 长度: {len(audio_data)/self.sample_rate:.2f}秒")
            
            # 如果音频文件太多，删除旧文件
            self.cleanup_debug_files(max_files=100)
            
            return debug_file
        except Exception as e:
            logger.warning(f"无法保存调试音频: {e}")
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
                    logger.debug(f"已删除旧的调试音频: {file_to_delete}")
        except Exception as e:
            logger.warning(f"清理调试文件时出错: {e}")

    def transcribe(self, audio_data, beam_size=3, language=None, translate=False):
        """
            转录音频数据
            audio_data: 音频数据，numpy数组
            beam_size: beam search大小
            language: 指定语言代码，如"zh"表示中文，"en"表示英文，None表示自动检测
            translate: 是否翻译为英文
        """

        try:
            # 添加音频数据的调试信息
            logger.info(f"开始转录音频: 长度={len(audio_data)} 样本, 最大值={np.max(np.abs(audio_data))}")
            
            # 检查音频数据是否有效
            if len(audio_data) < 1000:  # 太短的音频可能无法转录
                logger.warning("音频数据太短，可能无法正确转录")
                return "", None
                
            if np.max(np.abs(audio_data)) < 0.01:  # 音量太小
                logger.warning("音频音量太小，可能是静音")
                return "", None
                
            # 确保音频数据是float32类型
            if audio_data.dtype != np.float32:
                logger.warning(f"音频数据类型不是float32，而是{audio_data.dtype}，进行转换")
                audio_data = audio_data.astype(np.float32)

            # 确保音频数据在正确的范围内
            max_abs = np.max(np.abs(audio_data))
            if max_abs > 1.0:
                logger.warning(f"音频数据超出范围，最大值为{max_abs}，进行归一化")
                audio_data = audio_data / max_abs
            elif max_abs < 0.1:
                # 如果音频信号太弱，增强它
                logger.warning(f"音频信号太弱，最大值为{max_abs}，进行增强")
                gain = min(0.5 / max_abs if max_abs > 0 else 1.0, 10.0)
                audio_data = audio_data * gain
                logger.info(f"应用增益: {gain}, 新的最大值: {np.max(np.abs(audio_data))}")
                                            
            # 保存转录前的音频样本
            self.save_debug_audio(audio_data, prefix="asr_input")

            # 进行转录
            logger.info("调用Whisper模型进行转录...")

            segments, info = self.model.transcribe(
                audio_data, 
                beam_size=max(beam_size, 5), 
                language=language,
                task="translate" if translate else "transcribe"  # 添加翻译选项
            )
            
            # 收集所有文本片段
            segments_list = list(segments)  # 将生成器转换为列表以便多次使用
            
            if not segments_list:
                logger.warning("Whisper模型未返回任何文本片段")
                return "", info

            transcription = " ".join(segment.text for segment in segments_list)                

            # 添加转录结果的调试信息
            logger.info(f"转录完成: '{transcription}'")
            return transcription, info
        except Exception as e:
            logger.error(f"转录错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "", None
    
    
    def transcribe_segment(self, audio_data, max_duration_seconds=10, beam_size=3, language=None):
        """转录音频片段，限制最大处理长度"""
        try:
            # 使用最后N秒的音频进行转录，避免处理过长的音频
            last_n_seconds = min(self.sample_rate * max_duration_seconds, len(audio_data))
            recent_audio = audio_data[-last_n_seconds:]
            
            # 保存转录前的音频片段
            self.save_debug_audio(recent_audio, prefix="segment")

            segments, info = self.model.transcribe(recent_audio, beam_size=beam_size, language=language)
            transcription = " ".join(segment.text for segment in segments)
            
            return transcription, info
        except Exception as e:
            logger.error(f"片段转录错误: {e}")
            return "", None