<template>
  <div class="min-h-screen bg-gray-100 p-4">
    <div class="max-w-3xl mx-auto bg-white rounded-lg shadow-md p-6">
      <h1 class="text-2xl font-bold text-center text-gray-800 mb-6">Whisper 语音识别客户端</h1>

      <!-- 连接状态 -->
      <div class="mb-6">
        <div class="flex items-center justify-between mb-2">
          <span class="text-gray-700 font-medium">服务器连接状态:</span>
          <span :class="isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'"
            class="px-3 py-1 rounded-full text-sm font-medium">
            {{ isConnected ? '已连接' : '未连接' }}
          </span>
        </div>
        <div class="flex space-x-2">
          <input v-model="serverUrl" type="text"
            class="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            placeholder="WebSocket服务器地址" />
          <button @click="toggleConnection" class="px-4 py-2 rounded-md text-white font-medium"
            :class="isConnected ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-500 hover:bg-blue-600'">
            {{ isConnected ? '断开连接' : '连接' }}
          </button>
        </div>
      </div>

      <!-- 录音控制 -->
      <div class="mb-6">
        <div class="flex items-center justify-between mb-2">
          <span class="text-gray-700 font-medium">录音状态:</span>
          <span :class="isRecording ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'"
            class="px-3 py-1 rounded-full text-sm font-medium">
            {{ isRecording ? '录音中' : '未录音' }}
          </span>
        </div>
        <div class="flex justify-center">
          <button @click="toggleRecording"
            class="px-6 py-3 rounded-full text-white font-medium flex items-center justify-center"
            :class="isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'"
            :disabled="!isConnected">
            <span v-if="isRecording" class="mr-2">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <rect x="6" y="6" width="8" height="8" />
              </svg>
            </span>
            <span v-else class="mr-2">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <circle cx="10" cy="10" r="5" />
              </svg>
            </span>
            {{ isRecording ? '停止录音' : '开始录音' }}
          </button>
        </div>

        <!-- 音量指示器 -->
        <div class="mt-4">
          <div class="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div class="h-full bg-blue-500 transition-all duration-200"
              :style="{ width: `${Math.min(volume * 100 * 5, 100)}%` }"></div>
          </div>
          <div class="text-xs text-gray-500 mt-1 text-center">音量: {{ (volume * 100).toFixed(2) }}%</div>
        </div>

        <!-- 添加初始化调试文件系统按钮 -->
        <div class="mt-4 flex justify-center">
          <button @click="initDebugFileSystem"
            class="px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-md">
            初始化调试文件系统
          </button>
        </div>
      </div>

      <!-- 转录结果 -->
      <div class="mb-6">
        <h2 class="text-lg font-medium text-gray-700 mb-2">转录结果:</h2>
        <div class="bg-gray-50 rounded-md p-4 min-h-[100px] max-h-[200px] overflow-y-auto border border-gray-200">
          <p class="text-gray-800">{{ transcription || '等待转录...' }}</p>
        </div>
      </div>

      <!-- 状态信息 -->
      <div class="mb-6">
        <h2 class="text-lg font-medium text-gray-700 mb-2">状态信息:</h2>
        <div class="bg-gray-50 rounded-md p-4 min-h-[50px] border border-gray-200">
          <p class="text-gray-800">{{ status || '等待状态更新...' }}</p>
        </div>
      </div>

      <div>
        <div class="flex items-center justify-between mb-2">
          <h2 class="text-lg font-medium text-gray-700">日志信息:</h2>
          <button @click="clearLogs" class="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700">
            清除日志
          </button>
        </div>
        <div
          class="bg-gray-800 text-gray-200 rounded-md p-4 min-h-[150px] max-h-[300px] overflow-y-auto font-mono text-sm text-left">
          <div v-for="(log, index) in logs" :key="index" class="mb-1 flex items-start">
            <span class="text-gray-400 whitespace-nowrap inline-block min-w-[70px]">{{ log.time }}</span>
            <span :class="{
            'text-green-400': log.type === 'info',
            'text-yellow-400': log.type === 'warning',
            'text-red-400': log.type === 'error'
          }" class="ml-2 whitespace-nowrap inline-block min-w-[60px]">
              [{{ log.type.toUpperCase() }}]
            </span>
            <span class="ml-2 break-words flex-1 text-left">{{ log.message }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted } from 'vue';
import WebSocketService from './services/WebSocket';
import AudioService from './services/AudioService';

export default {
  name: 'App',
  setup() {
    const serverUrl = ref('ws://localhost:8765');
    const isConnected = ref(false);
    const isRecording = ref(false);
    const transcription = ref('');
    const status = ref('');
    const volume = ref(0);
    const logs = ref([]);

    // 添加日志
    // 添加日志
    const addLog = (type, message) => {
      const now = new Date();
      const timeString = now.toLocaleTimeString();
      logs.value.unshift({
        time: timeString,
        type,
        message
      });

      // 限制日志数量
      if (logs.value.length > 100) {
        logs.value = logs.value.slice(0, 100);
      }

      // 修复 console 方法调用
      switch (type) {
        case 'warning':
          console.warn(message);
          break;
        case 'error':
          console.error(message);
          break;
        case 'info':
        default:
          console.log(message);
          break;
      }
    };

    // 清除日志
    const clearLogs = () => {
      logs.value = [];
    };

    // 连接/断开WebSocket
    const toggleConnection = async () => {
      if (isConnected.value) {
        WebSocketService.disconnect();
        isConnected.value = false;
        addLog('info', '已断开与服务器的连接');
      } else {
        try {
          addLog('info', `正在连接到服务器: ${serverUrl.value}`);
          await WebSocketService.connect(serverUrl.value);
          isConnected.value = true;
          addLog('info', '已成功连接到服务器');
        } catch (error) {
          addLog('error', `连接服务器失败: ${error.message || '未知错误'}`);
        }
      }
    };

    // 设置音频服务回调
    const setupAudioCallbacks = () => {
      console.log('设置音频服务回调...');

      AudioService.setCallbacks({
        onAudioData: (audioData) => {
          try {
            console.log(`音频回调触发: 收到 ${audioData.length} 样本的音频数据`);

            if (isRecording.value && isConnected.value) {
              try {
                // 记录音频统计信息，帮助调试
                // 使用setTimeout避免同步调用栈过深
                setTimeout(() => {
                  try {
                    const stats = AudioService.logAudioStats(audioData);
                    if (stats && !stats.isSilent) {
                      addLog('info', `音频统计: RMS=${stats.rms.toFixed(4)}, 最大值=${stats.max.toFixed(4)}, 最小值=${stats.min.toFixed(4)}`);
                    }                    
                  } catch (statsError) {
                    console.error('记录音频统计信息失败:', statsError);
                    console.error('统计错误调用栈:', statsError.stack);
                  }
                }, 0);
              } catch (statsError) {
                console.error('记录音频统计信息失败:', statsError);
                console.error('统计错误调用栈:', statsError.stack);
              }

              // 将Float32Array转换为服务端可处理的格式
              let audioBlob;
              try {
                audioBlob = AudioService.prepareAudioForSending(audioData);
                if (!audioBlob) {
                  addLog('error', '音频数据准备失败');
                  return;
                }
              } catch (conversionError) {
                addLog('error', `音频转换失败: ${conversionError.message}`);
                console.error('转换错误调用栈:', conversionError.stack);
                return;
              }

              // 添加音频时长估计
              const durationSec = audioData.length / AudioService.targetSampleRate;
              addLog('info', `处理音频数据: ${audioData.length} 样本, 估计时长: ${durationSec.toFixed(2)}秒`);

              // 保存调试音频
              try {
                // 使用Promise.resolve().then()将操作放入微任务队列
                Promise.resolve().then(() => {
                  // 先将audioData转换为WAV格式
                  const wavBlob = AudioService.float32ToWav(audioData, AudioService.targetSampleRate);
                  return AudioService.saveDebugWavFile(wavBlob, 'send');
                })
                  .then(debugUrl => {
                    // 处理成功的情况
                    if (debugUrl) {
                      addLog('info', `已保存调试音频: ${debugUrl}`);
                    } else {
                      addLog('warning', '保存调试音频失败，但将继续发送');
                    }

                    // 发送音频数据到服务器
                    return sendAudioData(audioBlob, audioData.length, durationSec);
                  })
                  .catch(error => {
                    addLog('error', `保存调试音频失败: ${error.message}`);
                    console.error('保存错误调用栈:', error.stack);

                    // 即使保存失败也尝试发送
                    return sendAudioData(audioBlob, audioData.length, durationSec);
                  });
              } catch (saveError) {
                addLog('error', `保存调试音频过程中出错: ${saveError.message}`);
                console.error('保存过程错误调用栈:', saveError.stack);

                // 尝试直接发送
                sendAudioData(audioBlob, audioData.length, durationSec);
              }
            } else {
              console.log('忽略音频数据: 未录音或未连接');
            }
          } catch (error) {
            addLog('error', `处理音频数据时出错: ${error.message}`);
            console.error('处理音频数据错误调用栈:', error.stack);
          }
        },
        onSilenceDetected: () => {
          if (isRecording.value && isConnected.value) {
            addLog('info', '检测到静音，发送静音信号');
            WebSocketService.send('SILENCE_DETECTED');
          }
        },
        onVolumeChange: (newVolume) => {
          volume.value = newVolume;
        }
      });

      console.log('音频服务回调设置完成');
    };

    // 提取发送音频数据的功能为单独的函数
    const sendAudioData = (audioBlob, sampleCount, durationSec) => {
      try {
        addLog('info', `准备发送音频数据: ${sampleCount} 样本, WAV大小: ${audioBlob.size} 字节, 估计时长: ${durationSec.toFixed(2)}秒`);
        const success = WebSocketService.send(audioBlob);
        if (success) {
          addLog('info', `已发送音频数据: ${audioBlob.size} 字节, 采样率: ${AudioService.targetSampleRate}Hz, 估计时长: ${durationSec.toFixed(2)}秒`);
        } else {
          addLog('error', '发送音频数据失败');
        }
        return success;
      } catch (sendError) {
        addLog('error', `发送音频数据失败: ${sendError.message}`);
        console.error('发送错误调用栈:', sendError.stack);
        return false;
      }
    };

    // 修改 toggleRecording 函数，增加错误处理
    const toggleRecording = async () => {
      if (!isConnected.value) {
        addLog('warning', '请先连接到服务器');
        return;
      }

      if (isRecording.value) {
        try {
          // 先通知服务器录音即将停止
          if (isConnected.value) {
            addLog('info', '通知服务器录音停止');
            WebSocketService.send('RECORDING_STOPPED');
          }

          // 停止录音 - 这会处理并发送剩余的音频数据
          AudioService.stopRecording();
          isRecording.value = false;
          addLog('info', '已停止录音');
        } catch (error) {
          addLog('error', `停止录音时出错: ${error.message}`);
          // 确保录音状态正确
          isRecording.value = false;
          // 尝试清理资源
          try {
            if (AudioService.audioProcessor) {
              AudioService.audioProcessor.disconnect();
              AudioService.audioProcessor = null;
            }
            AudioService.isRecording = false;
            AudioService.audioQueue = [];
          } catch (cleanupError) {
            console.error('清理录音资源时出错:', cleanupError);
          }
        }
      } else {
        try {
          // 初始化音频
          if (!AudioService.audioStream) {
            addLog('info', '正在初始化音频...');
            const success = await AudioService.initAudio();
            if (!success) {
              addLog('error', '初始化音频失败');
              return;
            }
            addLog('info', '音频初始化成功');
          }

          // 确保回调已设置
          setupAudioCallbacks();

          // 通知服务器开始录音
          if (isConnected.value) {
            addLog('info', '通知服务器开始录音');
            WebSocketService.send('RECORDING_STARTED');
          }

          // 开始录音
          const success = AudioService.startRecording();
          if (success) {
            isRecording.value = true;
            addLog('info', '开始录音');
          } else {
            addLog('error', '开始录音失败');
          }
        } catch (error) {
          addLog('error', `开始录音时出错: ${error.message}`);
        }
      }
    };

    // 初始化调试文件系统
    const initDebugFileSystem = async () => {
      addLog('info', '正在初始化调试文件系统...');
      addLog('info', '请在弹出的对话框中选择一个目录，系统将在其中创建debug_browser_files文件夹用于保存调试音频');

      try {
        // 调用 AudioService 的初始化方法，传递 true 表示来自用户交互
        const success = await AudioService.autoInitFileSystemAccess(true);
        if (success) {
          addLog('info', '调试文件系统初始化成功，音频文件将保存到选定目录下的debug_browser_files文件夹中');
        } else {
          addLog('warning', '调试文件系统初始化失败，调试音频将通过浏览器下载功能保存');
        }
      } catch (error) {
        addLog('error', `初始化调试文件系统失败: ${error.message}`);
      }
    };

    // 修改setupWebSocketCallbacks函数
    const setupWebSocketCallbacks = () => {
      WebSocketService.setCallbacks({
        onMessage: (event) => {
          if (!(event.data instanceof Blob)) {
            addLog('info', `收到消息: ${event.data}`);
          } else {
            addLog('info', `收到音频数据: ${event.data.size} 字节`);
          }
        },
        onAudioData: async (audioBlob) => {
          addLog('info', `收到音频数据，准备播放: ${audioBlob.size} 字节`);

          try {
            // 通知服务器开始播放音频
            WebSocketService.send('playback_started');
            addLog('info', '已通知服务器开始播放音频');

            // 播放音频
            await AudioService.playAudio(audioBlob);
            addLog('info', '音频播放完成');

            // 通知服务器音频播放完成
            WebSocketService.send('playback_finished');
            addLog('info', '已通知服务器音频播放完成');
          } catch (error) {
            addLog('error', `音频播放失败: ${error.message || '未知错误'}`);

            // 即使播放失败，也通知服务器继续
            WebSocketService.send('playback_error');
            addLog('warning', '已通知服务器音频播放失败');
          }
        },
        onTranscription: (text) => {
          transcription.value = text;
          addLog('info', `收到转录: ${text}`);
        },
        onStatusChange: (newStatus) => {
          status.value = newStatus;
          addLog('info', `状态更新: ${newStatus}`);
        },
        onConnectionChange: (connected) => {
          isConnected.value = connected;
          if (!connected) {
            isRecording.value = false;
            AudioService.stopRecording();
            addLog('warning', '与服务器的连接已断开');
          }
        },
        onError: (error) => {
          addLog('error', `WebSocket错误: ${error.message || '未知错误'}`);
        }
      });
    };

    onMounted(() => {
      setupWebSocketCallbacks();
      setupAudioCallbacks();

      // 初始化音频
      AudioService.initAudio().then(() => {
        if (AudioService.actualSampleRate) {
          addLog('info', `浏览器实际采样率: ${AudioService.actualSampleRate}Hz, 目标采样率: ${AudioService.targetSampleRate}Hz`);

          if (AudioService.actualSampleRate !== AudioService.targetSampleRate) {
            addLog('info', '将自动进行音频重采样以匹配目标采样率');
          }
        }
      }).catch(error => {
        addLog('error', `初始化音频失败: ${error.message}`);
      });

      // 尝试初始化文件系统
      AudioService.autoInitFileSystemAccess().then(success => {
        if (success) {
          addLog('info', '调试文件系统初始化成功');
        } else {
          addLog('warning', '调试文件系统初始化失败，可以点击"初始化调试文件系统"按钮手动初始化');
        }
      }).catch(error => {
        addLog('error', `初始化文件系统失败: ${error.message}`);
      });
    });

    onUnmounted(() => {
      if (isConnected.value) {
        WebSocketService.disconnect();
      }

      if (isRecording.value) {
        AudioService.stopRecording();
      }

      // 释放音频资源
      if (AudioService.audioStream) {
        AudioService.audioStream.getTracks().forEach(track => track.stop());
      }
    });

    return {
      serverUrl,
      isConnected,
      isRecording,
      transcription,
      status,
      volume,
      logs,
      toggleConnection,
      toggleRecording,
      clearLogs,
      initDebugFileSystem
    };
  }
};
</script>

<style scoped>
.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}

.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}

.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}
</style>