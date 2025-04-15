// 在文件顶部添加导入
import AudioService from './AudioService';

class WebSocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectTimeout = null;
    this.reconnectDelay = 2000; // 初始重连延迟，单位毫秒
    this.callbacks = {
      onMessage: null,
      onConnect: null,
      onDisconnect: null,
      onError: null,
      onAudioData: null,
      onTranscription: null,
      onStatusChange: null
    };
    this.serverUrl = null;

    // 添加静音检测相关变量，匹配client.py
    this.lastActiveTime = Date.now(); // 上次检测到有效音频的时间
    this.isSilent = false; // 当前是否处于静音状态
    this.paused = false; // 录音是否已暂停
    this.consecutiveSilenceBlocks = 0; // 连续静音块计数
    this.silenceThreshold = 0.1; // 静音检测阈值
    this.silenceDuration = 5000; // 静音持续5秒后停止录音

    // 添加消息统计
    this.messageCount = 0;
    this.audioMessageCount = 0;
  }

  connect(url) {
    // 如果已经连接，先断开
    if (this.socket) {
      this.disconnect();
    }

    this.serverUrl = url;
    console.log(`尝试连接到WebSocket服务器: ${url}`);

    try {
      this.socket = new WebSocket(url);

      // 设置二进制类型为arraybuffer，而不是默认的blob
      this.socket.binaryType = 'arraybuffer';

      this.socket.onopen = () => {
        console.log('WebSocket连接已建立');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        if (this.callbacks.onConnect) {
          this.callbacks.onConnect();
        }
      };

      this.socket.onmessage = (event) => {
        try {
          this.messageCount++;

          // 检查数据类型
          if (event.data instanceof ArrayBuffer) {
            const size = event.data.byteLength;
            this.audioMessageCount++;
            console.log(`收到二进制数据: ${size} 字节 (消息 #${this.messageCount}, 音频 #${this.audioMessageCount})`);

            // 将ArrayBuffer转换为Blob以便于处理
            const audioBlob = new Blob([event.data], { type: 'audio/wav' });

            // 通知服务器开始播放音频
            this.send("playback_started").then(() => {
              console.log("playback_started 开始播放从服务端过来的音频");
            }).catch(error => {
              console.error("发送playback_started失败:", error);
            });

            // 如果有处理二进制数据的回调，则调用它
            if (this.callbacks.onAudioData) {
              this.callbacks.onAudioData(audioBlob, true); // 添加第二个参数表示这是服务器发送的音频
            }            
          } else {
            // 尝试解析JSON
            try {
              const message = JSON.parse(event.data);
              console.log('收到WebSocket消息:', message);
              if (this.callbacks.onMessage) {
                this.callbacks.onMessage(message);
              }
            } catch (parseError) {
              // 如果不是JSON，则作为普通文本处理
              console.log('收到WebSocket文本消息:', event.data);

              // 检查是否是转录结果
              if (typeof event.data === 'string' && event.data.startsWith('TRANSCRIPTION:')) {
                const transcription = event.data.substring('TRANSCRIPTION:'.length).trim();
                console.log('收到转录文本:', transcription);
                if (this.callbacks.onTranscription) {
                  this.callbacks.onTranscription(transcription);
                }
              }
              // 检查是否是状态更新
              else if (typeof event.data === 'string' && event.data.startsWith('STATUS:')) {
                const status = event.data.substring('STATUS:'.length).trim();
                if (this.callbacks.onStatusChange) {
                  this.callbacks.onStatusChange(status);
                }
              }
              // 其他文本消息
              else if (this.callbacks.onMessage) {
                this.callbacks.onMessage(event.data);
              }
            }
          }
        } catch (error) {
          console.error('解析WebSocket消息失败:', error);
        }
      };

      this.socket.onclose = (event) => {
        this.isConnected = false;
        console.log(`WebSocket连接已关闭: 代码=${event.code}, 原因=${event.reason}`);

        if (this.callbacks.onDisconnect) {
          this.callbacks.onDisconnect(event);
        }

        // 尝试重新连接
        this._attemptReconnect();
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket错误:', error);
        if (this.callbacks.onError) {
          this.callbacks.onError(error);
        }
      };

      return true;
    } catch (error) {
      console.error('创建WebSocket连接失败:', error);
      return false;
    }
  }

  disconnect() {
    if (this.socket) {
      console.log('正在关闭WebSocket连接...');
      this.socket.close();
      this.socket = null;
      this.isConnected = false;
    }

    // 清除重连定时器
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }

  sendMessage(message) {
    if (!this.isConnected) {
      console.error('WebSocket未连接，无法发送消息');
      return false;
    }

    try {
      const messageString = JSON.stringify(message);
      this.socket.send(messageString);
      return true;
    } catch (error) {
      console.error('发送WebSocket消息失败:', error);
      return false;
    }
  }

  async send(data) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.error('WebSocket未连接，无法发送数据');
      return false;
    }

    try {
      // 如果是字符串，直接发送
      if (typeof data === 'string') {
        this.socket.send(data);
        console.log(`已发送文本消息: ${data}`);
        return true;
      }
      
      // 如果是Blob或ArrayBuffer，直接发送
      if (data instanceof Blob || data instanceof ArrayBuffer) {
        this.socket.send(data);
        console.log(`已发送二进制数据: ${data instanceof Blob ? data.size : data.byteLength} 字节`);
        return true;
      }
      
      // 如果是对象，转换为JSON字符串
      if (typeof data === 'object') {
        const jsonStr = JSON.stringify(data);
        this.socket.send(jsonStr);
        console.log(`已发送JSON数据: ${jsonStr}`);
        return true;
      }
      
      console.error('不支持的数据类型:', typeof data);
      return false;
    } catch (error) {
      console.error('发送数据失败:', error);
      return false;
    }
  }

  sendBinaryData(data, mimeType = 'audio/wav') {
    if (!this.isConnected) {
      console.error('WebSocket未连接，无法发送二进制数据');
      return false;
    }

    try {
      // 确保数据是Blob类型
      let blobToSend;
      if (data instanceof Blob) {
        // 如果已经是Blob，确保MIME类型正确
        if (data.type !== mimeType) {
          blobToSend = new Blob([data], { type: mimeType });
        } else {
          blobToSend = data;
        }
      } else if (data instanceof ArrayBuffer) {
        // 如果是ArrayBuffer，转换为Blob
        blobToSend = new Blob([data], { type: mimeType });
      } else {
        console.error('不支持的数据类型，无法发送');
        return false;
      }

      // 添加调试日志
      console.log(`发送二进制数据: ${blobToSend.size} 字节, 类型: ${blobToSend.type}`);

      // 发送前检查数据完整性
      if (blobToSend.size <= 44) { // WAV头的大小是44字节
        console.error('数据太小，可能只有WAV头没有实际音频数据');
        return false;
      }

      // 发送数据
      this.socket.send(blobToSend);
      return true;
    } catch (error) {
      console.error('发送二进制数据失败:', error);
      return false;
    }
  }

  setCallbacks(callbacks) {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  _attemptReconnect() {
    // 如果已经达到最大重连次数，不再尝试
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log(`已达到最大重连尝试次数 (${this.maxReconnectAttempts})，停止重连`);
      return;
    }

    // 计算下一次重连延迟（指数退避）
    const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts);

    console.log(`将在 ${delay}ms 后尝试重新连接 (尝试 ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectAttempts++;
      this.connect(this.serverUrl);
    }, delay);
  }
}

export default new WebSocketService();