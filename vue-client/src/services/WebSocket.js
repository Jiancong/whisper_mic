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
      onError: null
    };
    this.serverUrl = null;
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
          // 检查数据类型
          if (event.data instanceof Blob) {
            console.log(`收到二进制数据: ${event.data.size} 字节`);
            // 如果有处理二进制数据的回调，则调用它
            if (this.callbacks.onAudioData) {
              this.callbacks.onAudioData(event.data);
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
              if (event.data.startsWith('TRANSCRIPTION:')) {
                const transcription = event.data.substring('TRANSCRIPTION:'.length).trim();
                if (this.callbacks.onTranscription) {
                  this.callbacks.onTranscription(transcription);
                }
              }
              // 检查是否是状态更新
              else if (event.data.startsWith('STATUS:')) {
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

  send(data) {
    if (!this.isConnected) {
      console.error('WebSocket未连接，无法发送数据');
      return false;
    }

    try {
      // 处理不同类型的数据
      if (data instanceof Blob) {
        console.log(`发送二进制数据: ${data.size} 字节, 类型: ${data.type}`);
        
        // 直接发送Blob数据，不尝试转换
        this.socket.send(data);
        return true;
      } else if (typeof data === 'string') {
        console.log(`发送文本消息: ${data.substring(0, 50)}${data.length > 50 ? '...' : ''}`);
        this.socket.send(data);
        return true;
      } else {
        // 对象类型，转为JSON
        const messageString = JSON.stringify(data);
        console.log(`发送JSON消息: ${messageString.substring(0, 50)}${messageString.length > 50 ? '...' : ''}`);
        this.socket.send(messageString);
        return true;
      }
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