// 在文件顶部添加导入
import AudioService from './AudioService';

class WebSocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.callbacks = {
      onMessage: null,
      onAudioData: null,
      onTranscription: null,
      onStatusChange: null,
      onConnectionChange: null,
      onError: null
    };
    this.debug = true; // 启用调试日志
  }

  connect(url) {
    return new Promise((resolve, reject) => {
      this.socket = new WebSocket(url);

      // 设置为blob类型，更适合处理音频数据
      this.socket.binaryType = 'blob';

      this.socket.onopen = () => {
        console.log('WebSocket连接已建立');
        this.isConnected = true;
        if (this.callbacks.onConnectionChange) {
          this.callbacks.onConnectionChange(true);
        }
        resolve();
      };

      this.socket.onclose = () => {
        console.log('WebSocket连接已关闭');
        this.isConnected = false;
        if (this.callbacks.onConnectionChange) {
          this.callbacks.onConnectionChange(false);
        }
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket错误:', error);
        if (this.callbacks.onError) {
          this.callbacks.onError(error);
        }
        reject(error);
      };

      this.socket.onmessage = (event) => {
        if (this.callbacks.onMessage) {
          this.callbacks.onMessage(event);
        }

        if (event.data instanceof Blob) {
          // 处理音频数据
          console.log(`收到二进制数据: ${event.data.size} 字节, 类型: ${event.data.type}`);

          // 检查Blob类型，确保是音频数据
          if (event.data.type === '' || event.data.type === 'application/octet-stream') {
            // 设置正确的MIME类型
            const audioBlob = new Blob([event.data], { type: 'audio/wav' });

            // 保存接收到的音频用于调试（会自动尝试初始化文件系统）
            if (AudioService && typeof AudioService.saveDebugWavFile === 'function') {
              console.log('尝试保存接收到的音频数据...');
              AudioService.saveDebugWavFile(audioBlob, 'receive')
                .then(debugUrl => {
                  console.log(`已保存接收到的音频数据，调试URL: ${debugUrl}`);
                })
                .catch(error => {
                  console.error('保存接收音频失败:', error);
                });
            }

            if (this.callbacks.onAudioData) {
              console.log('调用onAudioData回调处理音频数据');
              this.callbacks.onAudioData(audioBlob);
            }
          }
        } else {
          // 处理文本消息
          const message = event.data;
          console.log('收到消息:', message);

          if (message.startsWith('TRANSCRIPTION:')) {
            const transcription = message.substring(14).trim();
            if (this.callbacks.onTranscription) {
              this.callbacks.onTranscription(transcription);
            }
          } else if (message.startsWith('STATUS:')) {
            const status = message.substring(7).trim();
            if (this.callbacks.onStatusChange) {
              this.callbacks.onStatusChange(status);
            }
          } else if (this.callbacks.onMessage) {
            this.callbacks.onMessage(message);
          }
        }
      };
    });
  }

  disconnect() {
    if (this.socket && this.isConnected) {
      this.socket.close();
    }
  }

  send(data) {
    try {
      if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
        console.error('WebSocket未连接，无法发送数据');
        return false;
      }

      if (typeof data === 'string') {
        // 发送文本消息
        if (this.debug) console.log(`发送文本消息: ${data}`);
        this.socket.send(data);
      } else if (data instanceof Blob) {
        // 发送二进制数据
        if (this.debug) console.log(`发送二进制数据: ${data.size} 字节, 类型: ${data.type}`);
        this.socket.send(data);
      } else {
        console.error('不支持的数据类型:', typeof data);
        return false;
      }

      return true;
    } catch (error) {
      console.error('发送数据时出错:', error);
      console.error('发送错误调用栈:', error.stack);
      return false;
    }
  }

  setCallbacks(callbacks) {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }


  // 添加验证WAV头中采样率的方法
  async _verifyWavSampleRate(wavBlob) {
    try {
      // 只读取前44字节（WAV头）
      const headerBlob = wavBlob.slice(0, 44);
      const buffer = await headerBlob.arrayBuffer();
      const view = new DataView(buffer);

      // 检查RIFF标识
      const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
      if (riff !== 'RIFF') {
        return { verified: false, error: 'RIFF标识不正确' };
      }

      // 检查WAVE标识
      const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
      if (wave !== 'WAVE') {
        return { verified: false, error: 'WAVE标识不正确' };
      }

      // 读取采样率（字节24-27）
      const sampleRate = view.getUint32(24, true);

      // 验证采样率是否为16000
      if (sampleRate !== 16000) {
        return { verified: false, error: `采样率不是16000Hz (实际: ${sampleRate}Hz)` };
      }

      return { verified: true, sampleRate };
    } catch (error) {
      return { verified: false, error: `验证WAV头失败: ${error.message}` };
    }
  }
}

export default new WebSocketService();