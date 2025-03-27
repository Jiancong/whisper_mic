import numpy as np
import logging
from faster_whisper import WhisperModel
import librosa

# 配置日志
logger = logging.getLogger(__name__)

class ASRProcessor:
    def __init__(self, model_path="medium", compute_type="float16", device="cuda", sample_rate=16000):
        """初始化ASR处理器"""
        self.sample_rate = sample_rate
        logger.info(f"加载Whisper模型: {model_path}...")
        self.model = WhisperModel(model_path, compute_type=compute_type, device=device)
        logger.info(f"模型加载完成，使用设备: {device}")
    
    def denoise_audio(self, audio_data, noise_threshold=0.01):
        """对音频数据进行降噪处理"""
        try:
            stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
            magnitude, phase = librosa.magphase(stft)
            noise_frames = int(self.sample_rate * 0.1 / 512)
            noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            mask = magnitude > (noise_profile * noise_threshold)
            magnitude_denoised = magnitude * mask
            stft_denoised = magnitude_denoised * phase
            denoised_audio = librosa.istft(stft_denoised, hop_length=512, length=len(audio_data))
            return denoised_audio.astype(np.float32)
        except Exception as e:
            logger.error(f"降噪处理错误: {e}")
            return audio_data
    
    def transcribe(self, audio_data, beam_size=3, language=None):
        """转录音频数据"""
        try:
            segments, info = self.model.transcribe(audio_data, beam_size=beam_size, language=language)
            transcription = " ".join(segment.text for segment in segments)
            return transcription, info
        except Exception as e:
            logger.error(f"转录错误: {e}")
            return "", None
    
    def transcribe_segment(self, audio_data, max_duration_seconds=10, beam_size=3, language=None):
        """转录音频片段，限制最大处理长度"""
        try:
            # 使用最后N秒的音频进行转录，避免处理过长的音频
            last_n_seconds = min(self.sample_rate * max_duration_seconds, len(audio_data))
            recent_audio = audio_data[-last_n_seconds:]
            
            segments, info = self.model.transcribe(recent_audio, beam_size=beam_size, language=language)
            transcription = " ".join(segment.text for segment in segments)
            
            return transcription, info
        except Exception as e:
            logger.error(f"片段转录错误: {e}")
            return "", None