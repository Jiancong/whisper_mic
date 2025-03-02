import asyncio
import websockets
import soundfile as sf
import numpy as np
import librosa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_whisper():
    uri = "ws://localhost:8765"
    try:
        # 增加 ping_interval 和 ping_timeout
        async with websockets.connect(
            uri,
            ping_interval=20,  # 每 20 秒发送一次 ping
            ping_timeout=60    # 等待 60 秒超时
        ) as websocket:
            logger.info("✅ Connected to Whisper WebSocket Server!")
            # 读取音频文件
            data, samplerate = sf.read("./CTS-CN-F2F-2019-11-15-3.wav", dtype="float32")
            if samplerate != 16000:
                logger.warning(f"⚠️ Original samplerate {samplerate} != 16000, resampling to 16000 Hz.")
                data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                samplerate = 16000
            
            chunk_size = 80000  # 5 seconds at 16000 Hz
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                await websocket.send(chunk.tobytes())
                logger.info(f"Sent chunk {i // chunk_size + 1} of {len(data) // chunk_size + 1}, size: {len(chunk)} samples")
                await asyncio.sleep(0.1)  # 模拟实时流
                
                # 接收转录结果
                response = await websocket.recv()
                if response.startswith("Error:"):
                    logger.error(f"⚠️ Server error: {response}")
                else:
                    logger.info(f"🎤 Transcription: {response}")
    except ConnectionRefusedError:
        logger.error("❌ Failed to connect to server. Is it running?")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")

# 运行测试
asyncio.run(test_whisper())