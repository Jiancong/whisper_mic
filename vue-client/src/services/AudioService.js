class AudioService {

  constructor() {
    this.mediaRecorder = null;
    this.audioContext = null;
    this.audioStream = null;
    this.isRecording = false;
    this.silenceDetector = null;
    this.audioProcessor = null;
    this.silenceThreshold = 0.01;
    this.silenceDuration = 5000; // 5秒
    this.consecutiveSilenceTime = 0;
    this.lastAudioTime = 0;
    this.audioQueue = [];
    this.processingInterval = null;
    this.callbacks = {
      onAudioData: null,
      onSilenceDetected: null,
      onVolumeChange: null
    };
    // 添加新的属性
    this.paused = false; // 是否暂停录音
    this.consecutiveSilenceBlocks = 0; // 连续静音块计数
    this.minAudioDuration = 5000; // 最小音频片段时长(毫秒)
    this.accumulatedSamples = 0; // 累积的样本数
    this.lastActiveTime = Date.now(); // 上次检测到活动音频的时间
    this.fileSystemAccessAttempted = false; // 是否已尝试访问文件系统
    this.audioBufferSize = 8192 * 2; // 音频缓冲区大小
    this.targetSampleRate = 16000; // 目标采样率
    this.actualSampleRate = null; // 实际采样率，将在初始化时设置
    this.fileSystemInitialized = false; // 添加新标志，表示文件系统是否已成功初始化
    this.debugDirHandle = null; // 确保变量已定义    
  }

  async initAudio() {
    try {
      // 创建音频上下文，不指定采样率，使用浏览器默认值
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

      // 记录实际采样率
      this.actualSampleRate = this.audioContext.sampleRate;
      console.log(`浏览器实际采样率: ${this.actualSampleRate}Hz, 目标采样率: ${this.targetSampleRate}Hz`);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      this.audioStream = stream;
      return true;
    } catch (error) {
      console.error('初始化音频失败:', error);
      return false;
    }
  }

  // 优化重采样方法，避免栈溢出
  resampleAudio(audioData, fromSampleRate, toSampleRate) {

    if (fromSampleRate === toSampleRate) {
      return audioData; // 如果采样率相同，无需重采样
    }

    console.log(`执行重采样: 从 ${fromSampleRate}Hz 到 ${toSampleRate}Hz, 样本数: ${audioData.length}`);

    try {
      // 计算重采样后的长度
      const ratio = toSampleRate / fromSampleRate;
      const newLength = Math.round(audioData.length * ratio);

      // 检查新长度是否合理，防止内存溢出
      if (newLength <= 0 || newLength > 100000000) { // 设置一个合理的上限
        console.error(`重采样后的长度不合理: ${newLength}，使用原始数据`);
        return audioData;
      }

      const result = new Float32Array(newLength);

      // 使用分块处理的方式进行重采样，避免栈溢出
      const CHUNK_SIZE = 10000; // 每次处理的样本数
      let offsetResult = 0;
      let offsetAudio = 0;

      while (offsetResult < result.length) {
        const endOffset = Math.min(offsetResult + CHUNK_SIZE, result.length);

        // 处理当前块
        for (let i = offsetResult; i < endOffset; i++) {
          const indexAudio = Math.floor(offsetAudio); // 使用Math.floor代替位运算
          const alpha = offsetAudio - indexAudio;

          // 确保不会越界
          if (indexAudio >= audioData.length - 1) {
            result[i] = audioData[audioData.length - 1];
          } else {
            // 线性插值
            result[i] = audioData[indexAudio] * (1 - alpha) + audioData[indexAudio + 1] * alpha;
          }

          // 更新音频偏移量
          offsetAudio += 1 / ratio;
        }

        // 更新结果偏移量
        offsetResult = endOffset;
      }

      console.log(`重采样完成: 新样本数: ${result.length}`);
      return result;
    } catch (error) {
      console.error(`重采样过程中出错: ${error.message}`, error);
      // 出错时返回原始数据
      return audioData;
    }
  }

  // 优化停止录音方法，避免处理过大的数据导致栈溢出
  stopRecording() {
    if (!this.isRecording) return;

    console.log('停止录音，处理剩余音频数据...');

    try {
      // 处理队列中剩余的音频数据
      if (this.audioQueue.length > 0) {
        // 计算队列中所有音频数据的总长度
        const totalLength = this.audioQueue.reduce((acc, curr) => acc + curr.length, 0);
        console.log(`处理剩余的 ${this.audioQueue.length} 块音频数据，共 ${totalLength} 样本`);

        // 检查数据量是否过大
        if (totalLength > 1000000) { // 如果超过100万个样本，只处理最后一部分
          console.warn(`音频数据过大 (${totalLength} 样本)，只处理最后部分以避免栈溢出`);

          // 只保留最后几个块，总计不超过50万个样本
          let accumulatedLength = 0;
          let startIndex = this.audioQueue.length - 1;

          while (startIndex >= 0 && accumulatedLength < 500000) {
            accumulatedLength += this.audioQueue[startIndex].length;
            startIndex--;
          }

          // 调整起始索引
          startIndex = Math.max(0, startIndex + 1);

          // 创建新的队列
          const trimmedQueue = this.audioQueue.slice(startIndex);
          console.log(`裁剪后的队列: ${trimmedQueue.length} 块，约 ${accumulatedLength} 样本`);

          try {
            // 合并裁剪后的音频数据
            const mergedData = new Float32Array(accumulatedLength);
            let offset = 0;
            for (const audioData of trimmedQueue) {
              mergedData.set(audioData, offset);
              offset += audioData.length;
            }

            // 发送合并后的音频数据
            if (this.callbacks.onAudioData) {
              console.log(`发送最后的音频数据到回调: ${mergedData.length} 样本, 估计时长: ${(mergedData.length / this.targetSampleRate).toFixed(2)}秒`);

              // 使用setTimeout避免栈溢出
              setTimeout(() => {
                try {
                  this.callbacks.onAudioData(mergedData);
                } catch (callbackError) {
                  console.error('回调执行出错:', callbackError);
                  console.error('错误调用栈:', callbackError.stack);
                }
              }, 0);
            }
          } catch (mergeError) {
            console.error('合并音频数据出错:', mergeError);
            console.error('错误调用栈:', mergeError.stack);
          }
        } else {
          // 数据量合理，正常处理
          try {
            // 合并所有音频数据
            const mergedData = new Float32Array(totalLength);
            let offset = 0;
            for (const audioData of this.audioQueue) {
              mergedData.set(audioData, offset);
              offset += audioData.length;
            }

            // 发送合并后的音频数据
            if (this.callbacks.onAudioData) {
              console.log(`发送最后的音频数据到回调: ${mergedData.length} 样本, 估计时长: ${(mergedData.length / this.targetSampleRate).toFixed(2)}秒`);

              // 使用setTimeout避免栈溢出
              setTimeout(() => {
                try {
                  this.callbacks.onAudioData(mergedData);
                } catch (callbackError) {
                  console.error('回调执行出错:', callbackError);
                  console.error('错误调用栈:', callbackError.stack);
                }
              }, 0);
            }
          } catch (mergeError) {
            console.error('合并音频数据出错:', mergeError);
            console.error('错误调用栈:', mergeError.stack);
          }
        }
      } else {
        console.log('没有剩余的音频数据需要处理');
      }
    } catch (error) {
      console.error(`处理剩余音频数据时出错: ${error.message}`, error);
    } finally {
      // 无论如何都要清理资源
      this.isRecording = false;

      if (this.audioProcessor) {
        this.audioProcessor.disconnect();
        this.audioProcessor = null;
      }

      if (this.processingInterval) {
        clearInterval(this.processingInterval);
        this.processingInterval = null;
      }

      this.audioQueue = [];
    }
  }

  startRecording() {
    if (!this.audioStream || this.isRecording) return false;

    try {
      this.isRecording = true;
      this.consecutiveSilenceTime = 0;
      this.lastAudioTime = Date.now();
      this.paused = false;
      this.consecutiveSilenceBlocks = 0;
      this.accumulatedSamples = 0;
      this.lastActiveTime = Date.now();
      this.audioQueue = []; // 确保队列为空

      // 创建音频处理节点
      const source = this.audioContext.createMediaStreamSource(this.audioStream);
      this.audioProcessor = this.audioContext.createScriptProcessor(this.audioBufferSize, 1, 1);

      // 连接节点
      source.connect(this.audioProcessor);
      this.audioProcessor.connect(this.audioContext.destination);

      // 处理音频数据
      this.audioProcessor.onaudioprocess = (e) => {
        if (!this.isRecording) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const audioData = new Float32Array(inputData);

        // 添加调试日志，确认音频处理器正在接收数据
        console.log(`接收到音频数据: ${audioData.length} 样本`);

        // 检查音频数据是否有效
        let isValid = true;
        for (let i = 0; i < audioData.length; i++) {
          if (isNaN(audioData[i]) || !isFinite(audioData[i])) {
            isValid = false;
            break;
          }
        }

        if (!isValid) {
          console.warn('检测到无效的音频数据，跳过此帧');
          return;
        }

        // 计算音量
        const volume = this.calculateVolume(audioData);

        if (this.callbacks.onVolumeChange) {
          this.callbacks.onVolumeChange(volume);
        }

        // 添加调试日志，显示当前音量
        console.log(`当前音量: ${volume.toFixed(6)}, 静音阈值: ${this.silenceThreshold}`);

        // 检测静音 - 改进的静音检测逻辑
        const isSilent = volume < this.silenceThreshold;
        const currentTime = Date.now();
        const blockDuration = (audioData.length / this.audioContext.sampleRate) * 1000;

        // 无论是否静音，都将音频数据添加到队列中
        this.audioQueue.push(audioData);
        this.accumulatedSamples += audioData.length;

        // 添加调试日志，显示队列状态
        const totalLength = this.audioQueue.reduce((acc, curr) => acc + curr.length, 0);
        const queueDuration = (totalLength / this.audioContext.sampleRate) * 1000;
        console.log(`当前队列状态: ${this.audioQueue.length} 块, ${totalLength} 样本, 估计时长: ${queueDuration.toFixed(2)}ms`);

        if (isSilent) {
          this.consecutiveSilenceBlocks++;
          console.log(`检测到静音块 #${this.consecutiveSilenceBlocks}, 音量: ${volume.toFixed(6)}`);

          // 计算连续静音时间
          const silenceTime = this.consecutiveSilenceBlocks * blockDuration;

          // 如果连续静音时间超过阈值且录音未暂停
          if (silenceTime >= this.silenceDuration && !this.paused) {
            console.log(`检测到连续静音超过 ${this.silenceDuration}ms，暂停录音`);
            this.paused = true;

            // 发送静音检测事件
            if (this.callbacks.onSilenceDetected) {
              this.callbacks.onSilenceDetected();
            }

            // 如果队列中有足够的数据，强制处理一次
            if (this.audioQueue.length > 0 && queueDuration >= 1000) {
              console.log(`静音触发，处理队列中的 ${this.audioQueue.length} 块音频数据`);

              const mergedData = new Float32Array(totalLength);

              let offset = 0;
              for (const audioData of this.audioQueue) {
                mergedData.set(audioData, offset);
                offset += audioData.length;
              }

              // 重采样到目标采样率
              const resampledData = this.resampleAudio(mergedData, this.actualSampleRate, this.targetSampleRate);

              // 发送合并后的音频数据
              if (this.callbacks.onAudioData) {
                console.log(`发送音频数据到回调: ${resampledData.length} 样本, 估计时长: ${(resampledData.length / this.targetSampleRate).toFixed(2)}秒`);
                this.callbacks.onAudioData(resampledData);
              } else {
                console.warn('onAudioData 回调未设置，无法发送音频数据');
              }

              // 清空队列
              this.audioQueue = [];
              this.accumulatedSamples = 0;
            }
          }
        } else {
          // 如果当前块不是静音
          if (this.paused) {
            console.log('检测到活动音频，恢复录音');
            this.paused = false;
          }

          this.consecutiveSilenceBlocks = 0;
          this.lastActiveTime = currentTime;
        }
      };

      // 启动音频处理
      this.startAudioProcessing();

      console.log('录音已开始，音频处理器已设置');
      return true;
    } catch (error) {
      console.error('开始录音失败:', error);
      this.isRecording = false;
      return false;
    }
  }

  // 优化 startAudioProcessing 方法，避免栈溢出
  startAudioProcessing() {
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
    }

    console.log('开始音频处理...');
    
    // 使用较长的间隔时间，减少处理频率
    const PROCESSING_INTERVAL = 1000; // 1秒处理一次
    
    this.processingInterval = setInterval(() => {
      try {
        if (!this.isRecording || this.paused) {
          return;
        }

        // 检查队列是否为空
        if (this.audioQueue.length === 0) {
          console.log('音频队列为空，等待数据...');
          return;
        }

        // 计算队列中的总样本数
        const totalSamples = this.audioQueue.reduce((acc, curr) => acc + curr.length, 0);
        
        // 如果累积的样本数太少，等待更多数据
        const minSamples = this.targetSampleRate * 1; // 至少1秒的音频
        if (totalSamples < minSamples) {
          console.log(`累积的样本数 (${totalSamples}) 不足 ${minSamples}，等待更多数据...`);
          return;
        }

        console.log(`处理音频队列: ${this.audioQueue.length} 块, 共 ${totalSamples} 样本`);

        // 合并音频数据
        const mergedData = new Float32Array(totalSamples);
        let offset = 0;
        
        // 使用循环而不是forEach，避免回调函数开销
        for (let i = 0; i < this.audioQueue.length; i++) {
          const audioData = this.audioQueue[i];
          mergedData.set(audioData, offset);
          offset += audioData.length;
        }

        // 清空队列
        this.audioQueue = [];

        // 重要：添加重采样步骤
        const resampledData = this.resampleAudio(mergedData, this.actualSampleRate, this.targetSampleRate);
        console.log(`重采样结果: 从 ${mergedData.length} 样本 (${this.actualSampleRate}Hz) 到 ${resampledData.length} 样本 (${this.targetSampleRate}Hz)`);        

        // 使用setTimeout将回调放入事件队列，避免同步调用栈过深
        setTimeout(() => {
          try {
            if (this.callbacks.onAudioData) {
              // 修复：使用重采样后的数据而不是原始数据
              this.callbacks.onAudioData(resampledData);
            }
          } catch (callbackError) {
            console.error('音频数据回调执行出错:', callbackError);
            console.error('回调错误调用栈:', callbackError.stack);
          }
        }, 0);
      } catch (error) {
        console.error('音频处理出错:', error);
        console.error('处理错误调用栈:', error.stack);
      }
    }, PROCESSING_INTERVAL);
  }


  calculateVolume(audioData) {
    // 计算音频数据的平均绝对值作为音量
    let sum = 0;
    let validSamples = 0;

    for (let i = 0; i < audioData.length; i++) {
      const sample = audioData[i];
      if (!isNaN(sample) && isFinite(sample)) {
        sum += Math.abs(sample);
        validSamples++;
      }
    }

    return validSamples > 0 ? sum / validSamples : 0;
  }

  // 修改playAudio方法
  async playAudio(audioBlob) {
    return new Promise((resolve, reject) => {
      try {
        console.log(`准备播放音频: ${audioBlob.size} 字节, 类型: ${audioBlob.type}`);

        // 确保Blob有正确的MIME类型
        const audioWithCorrectType = audioBlob.type ?
          audioBlob :
          new Blob([audioBlob], { type: 'audio/wav' });

        // 创建URL
        const audioUrl = URL.createObjectURL(audioWithCorrectType);

        // 创建音频元素
        const audio = new Audio();

        // 添加事件监听器
        audio.oncanplaythrough = () => {
          console.log('音频已加载，准备播放');
        };

        audio.onplay = () => {
          console.log('音频开始播放');
          // 确保音频上下文已经启动
          if (this.audioContext && this.audioContext.state !== 'running') {
            this.audioContext.resume();
          }
        };

        audio.onended = () => {
          console.log('音频播放完成');
          URL.revokeObjectURL(audioUrl); // 释放URL
          resolve();
        };

        audio.onerror = (error) => {
          console.error('播放音频失败:', error);
          URL.revokeObjectURL(audioUrl); // 释放URL
          reject(error);
        };

        // 设置音频源并播放
        audio.src = audioUrl;

        // 确保音量足够大
        audio.volume = 1.0;

        // 确保播放速率正确
        audio.playbackRate = 1.0;

        // 播放音频
        audio.play().catch(error => {
          console.error('播放音频时出错:', error);
          reject(error);
        });
      } catch (error) {
        console.error('处理音频时出错:', error);
        reject(error);
      }
    });
  }

  setCallbacks(callbacks) {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }



  // 修改 saveDebugWavFile 方法，确保文件保存到正确的目录
  async saveDebugWavFile(wavBlob, prefix = 'debug') {
    try {
      // 检查文件系统是否已初始化
      if (!this.fileSystemInitialized || !this.debugDirHandle) {
        console.warn('调试文件系统未初始化，尝试重新初始化...');
        const result = await this.autoInitFileSystemAccess(false);
        if (result && result.needsUserGesture) {
          console.error('需要用户交互才能初始化文件系统，使用下载方式保存');
          return this._saveDebugWavFileDownload(wavBlob, prefix);
        }

        if (!result || !this.fileSystemInitialized) {
          console.error('无法初始化文件系统，将使用下载方式保存文件');
          return this._saveDebugWavFileDownload(wavBlob, prefix);
        }
      }

      // 生成文件名
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const fileName = `${prefix}_${timestamp}.wav`;

      console.log(`尝试保存调试文件到 debug_browser_files/${fileName}`);

      try {
        // 使用文件系统API保存文件
        const fileHandle = await this.debugDirHandle.getFileHandle(fileName, { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(wavBlob);
        await writable.close();

        console.log(`成功保存调试文件: debug_browser_files/${fileName}`);

        // 返回文件URL（这只是一个参考路径，不是真实的URL）
        return `debug_browser_files/${fileName}`;
      } catch (fsError) {
        console.error('使用文件系统API保存文件失败:', fsError);
        console.error('错误调用栈:', fsError.stack);

        // 回退到下载方式
        return this._saveDebugWavFileDownload(wavBlob, prefix);
      }
    } catch (error) {
      console.error('保存调试WAV文件失败:', error);
      console.error('错误调用栈:', error.stack);
      return null;
    }
  }

  // 添加下载方式保存文件的备用方法
  _saveDebugWavFileDownload(wavBlob, prefix = 'debug') {
    try {
      // 生成文件名
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const fileName = `${prefix}_${timestamp}.wav`;

      console.log(`使用下载方式保存调试文件: ${fileName}`);

      // 创建下载链接
      const url = URL.createObjectURL(wavBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;

      // 添加到DOM并触发点击
      document.body.appendChild(a);
      a.click();

      // 清理
      setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }, 100);

      return `下载/${fileName}`;
    } catch (error) {
      console.error('下载方式保存文件失败:', error);
      return null;
    }
  }

  // 修改文件系统初始化方法，确保正确创建目录
  async autoInitFileSystemAccess(fromUserGesture = false) {
    // 如果已经初始化过，直接返回结果
    if (this.fileSystemInitialized && this.debugDirHandle) {
      console.log('文件系统已初始化');
      return true;
    }

    // 如果已经尝试过但失败了，不再重复尝试
    if (this.fileSystemAccessAttempted && !this.fileSystemInitialized  && !fromUserGesture) {
      console.warn('之前已尝试初始化文件系统但失败，不再重试');
      return false;
    }

    this.fileSystemAccessAttempted = true;

    try {
      // 检查File System Access API是否可用
      if (!window.showDirectoryPicker) {
        console.warn('此浏览器不支持File System Access API');
        return false;
      }

      console.log('尝试获取文件系统访问权限...');

      // 请求用户选择目录
      const dirHandle = await window.showDirectoryPicker({
        id: 'whisperDebugDir',
        mode: 'readwrite',
        startIn: 'documents'
      });

      console.log('用户已选择目录:', dirHandle.name);

      // 尝试创建或获取debug_browser_files子目录
      try {
        this.debugDirHandle = await dirHandle.getDirectoryHandle('debug_browser_files', { create: true });
        console.log('成功创建/获取debug_browser_files子目录');
      } catch (subDirError) {
        console.error('创建debug_browser_files子目录失败:', subDirError);
        // 如果无法创建子目录，使用主目录
        this.debugDirHandle = dirHandle;
        console.log('将使用主目录保存调试文件');
      }

      // 验证写入权限
      try {
        // 创建一个测试文件
        const testFileName = `test_${Date.now()}.txt`;
        const testFileHandle = await this.debugDirHandle.getFileHandle(testFileName, { create: true });
        const writable = await testFileHandle.createWritable();
        await writable.write('测试文件系统访问权限');
        await writable.close();

        console.log('成功验证文件系统写入权限');

        // 尝试删除测试文件
        try {
          await this.debugDirHandle.removeEntry(testFileName);
          console.log('成功删除测试文件');
        } catch (removeError) {
          console.warn('无法删除测试文件，但这不影响功能:', removeError);
        }
      } catch (verifyError) {
        console.error('验证文件系统写入权限失败:', verifyError);
        return false;
      }

      this.fileSystemInitialized = true;
      console.log('文件系统访问初始化成功');
      return true;
    } catch (error) {
      console.error('初始化文件系统访问失败:', error);
      console.error('错误调用栈:', error.stack);
      this.fileSystemInitialized = false;
      return false;
    }
  }

  // 修改 float32ToWav 方法，添加调用栈跟踪
  float32ToWav(float32Array) {
    try {
      console.log(`转换Float32Array到WAV: ${float32Array.length} 样本, 采样率: ${this.targetSampleRate}Hz`);

      // 检查数据有效性
      if (!float32Array || float32Array.length === 0) {
        console.error('无效的音频数据: 空或长度为0');
        return null;
      }

      // 检查是否有NaN或Infinity值
      let hasInvalidValues = false;
      let minValue = Infinity;
      let maxValue = -Infinity;
      let sumValue = 0;

      for (let i = 0; i < float32Array.length; i++) {
        if (isNaN(float32Array[i]) || !isFinite(float32Array[i])) {
          hasInvalidValues = true;
          float32Array[i] = 0; // 将无效值替换为0
        } else {
          minValue = Math.min(minValue, float32Array[i]);
          maxValue = Math.max(maxValue, float32Array[i]);
          sumValue += Math.abs(float32Array[i]);
        }
      }

      const avgValue = sumValue / float32Array.length;      

      if (hasInvalidValues) {
        console.warn('检测到无效的音频样本值，已替换为0');
      }

      console.log(`音频数据统计: 最小值=${minValue.toFixed(6)}, 最大值=${maxValue.toFixed(6)}, 平均绝对值=${avgValue.toFixed(6)}`);

      // 如果音频数据过于安静，可能是录音问题
      if (avgValue < 0.001) {
        console.warn('音频数据非常安静，可能没有正确录制声音');
      }      

      // 创建WAV文件头
      const numChannels = 1; // 单声道
      const bitsPerSample = 16; // 16位PCM
      const bytesPerSample = bitsPerSample / 8;
      const blockAlign = numChannels * bytesPerSample;
      const byteRate = this.targetSampleRate * blockAlign;
      const dataSize = float32Array.length * bytesPerSample;
      const bufferSize = 44 + dataSize;

      // 安全检查
      if (bufferSize <= 44 || bufferSize > 100000000) {
        console.error(`无效的缓冲区大小: ${bufferSize}`);
        return null;
      }

      const buffer = new ArrayBuffer(bufferSize);
      const view = new DataView(buffer);

      // 写入WAV头
      // "RIFF"
      view.setUint8(0, 0x52);
      view.setUint8(1, 0x49);
      view.setUint8(2, 0x46);
      view.setUint8(3, 0x46);
      // 文件大小
      view.setUint32(4, 36 + dataSize, true);
      // "WAVE"
      view.setUint8(8, 0x57);
      view.setUint8(9, 0x41);
      view.setUint8(10, 0x56);
      view.setUint8(11, 0x45);
      // "fmt "
      view.setUint8(12, 0x66);
      view.setUint8(13, 0x6D);
      view.setUint8(14, 0x74);
      view.setUint8(15, 0x20);
      // 格式块大小
      view.setUint32(16, 16, true);
      // 音频格式 (1 = PCM)
      view.setUint16(20, 1, true);
      // 声道数
      view.setUint16(22, numChannels, true);
      // 采样率
      view.setUint32(24, this.targetSampleRate, true);
      // 字节率
      view.setUint32(28, byteRate, true);
      // 块对齐
      view.setUint16(32, blockAlign, true);
      // 每个样本的位数
      view.setUint16(34, bitsPerSample, true);
      // "data"
      view.setUint8(36, 0x64);
      view.setUint8(37, 0x61);
      view.setUint8(38, 0x74);
      view.setUint8(39, 0x61);
      // 数据大小
      view.setUint32(40, dataSize, true);

      // 写入音频数据
      let offsetData = 44;
      for (let i = 0; i < float32Array.length; i++) {
        // 将float32转换为int16
        const sample = Math.max(-1, Math.min(1, float32Array[i]));
        const value = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(offsetData, value, true);
        offsetData += 2;
      }

      // 创建Blob
      const wavBlob = new Blob([buffer], { type: 'audio/wav' });
      console.log(`WAV转换完成: ${wavBlob.size} 字节`);

      return wavBlob;

    } catch (error) {
      console.error(`Float32Array转WAV失败: ${error.message}`, error);
      console.error('错误调用栈:', error.stack);
      return null;
    }
  }

  // 辅助函数：将字符串写入DataView
  _writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // 优化 logAudioStats 方法，避免栈溢出
  logAudioStats(audioData) {
    try {
      // 检查数据有效性
      if (!audioData || audioData.length === 0) {
        console.log('音频数据为空，无法记录统计信息');
        return;
      }

      // 限制处理的样本数量，避免过度消耗栈空间
      const MAX_SAMPLES_TO_ANALYZE = 10000;
      const samplesToAnalyze = Math.min(audioData.length, MAX_SAMPLES_TO_ANALYZE);
      
      // 使用循环而不是数组方法来计算统计数据
      let min = Infinity;
      let max = -Infinity;
      let sum = 0;
      let sumSquares = 0;
      
      for (let i = 0; i < samplesToAnalyze; i++) {
        const value = audioData[i];
        
        // 跳过无效值
        if (isNaN(value) || !isFinite(value)) continue;
        
        min = Math.min(min, value);
        max = Math.max(max, value);
        sum += value;
        sumSquares += value * value;
      }
      
      // 计算统计值
      const mean = sum / samplesToAnalyze;
      const variance = (sumSquares / samplesToAnalyze) - (mean * mean);
      const rms = Math.sqrt(sumSquares / samplesToAnalyze);
      
      // 检测是否为静音
      const isSilent = rms < this.silenceThreshold;
      
      // 记录统计信息
      console.log(`音频统计 (${samplesToAnalyze}/${audioData.length} 样本): 
        最小值: ${min.toFixed(6)}, 
        最大值: ${max.toFixed(6)}, 
        平均值: ${mean.toFixed(6)}, 
        RMS: ${rms.toFixed(6)}, 
        是否静音: ${isSilent}`);
      
      return {
        min,
        max,
        mean,
        rms,
        isSilent
      };
    } catch (error) {
      console.error('记录音频统计信息失败:', error);
      console.error('错误调用栈:', error.stack);
      return null;
    }
  }



  // 添加保存到IndexedDB的辅助方法
  _saveToIndexedDB(wavBlob, fileName) {
    // 打开或创建IndexedDB数据库
    const request = indexedDB.open('AudioDebugDB', 1);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('audioFiles')) {
        db.createObjectStore('audioFiles', { keyPath: 'fileName' });
      }
    };

    request.onsuccess = (event) => {
      const db = event.target.result;
      const transaction = db.transaction(['audioFiles'], 'readwrite');
      const store = transaction.objectStore('audioFiles');

      // 保存Blob到IndexedDB
      store.put({
        fileName: fileName,
        data: wavBlob,
        timestamp: new Date().getTime()
      });

      console.log(`已保存调试音频到IndexedDB: ${fileName}`);

      // 清理旧文件（保留最近50个）
      const cleanupTransaction = db.transaction(['audioFiles'], 'readwrite');
      const cleanupStore = cleanupTransaction.objectStore('audioFiles');
      const getAllRequest = cleanupStore.getAll();

      getAllRequest.onsuccess = () => {
        const allFiles = getAllRequest.result;
        if (allFiles.length > 50) {
          // 按时间戳排序
          allFiles.sort((a, b) => b.timestamp - a.timestamp);

          // 删除旧文件
          for (let i = 50; i < allFiles.length; i++) {
            cleanupStore.delete(allFiles[i].fileName);
          }
        }
      };
    };

    request.onerror = (event) => {
      console.error('保存调试音频到IndexedDB失败:', event.target.error);
    };
  }

  // 修改保存到本地文件系统的辅助方法，增强错误处理
  async _saveToLocalFileSystem(wavBlob, fileName) {
    try {
      // 检查是否已经获取了目录句柄
      if (!this.debugDirHandle) {
        throw new Error('未初始化文件系统访问权限，无法保存到本地文件系统');
      }

      console.log(`尝试保存文件到本地文件系统: ${fileName}`);

      // 尝试创建或获取debug_browser_files子目录
      let debugDirHandle;
      try {
        debugDirHandle = await this.debugDirHandle.getDirectoryHandle('debug_browser_files', { create: true });
        console.log('已获取debug_browser_files子目录');
      } catch (dirError) {
        console.warn('获取debug_browser_files子目录失败，将直接使用选择的目录:', dirError);
        debugDirHandle = this.debugDirHandle;
      }

      // 创建文件
      console.log(`创建文件: ${fileName}`);
      const fileHandle = await debugDirHandle.getFileHandle(fileName, { create: true });

      // 获取可写流
      console.log('获取文件写入流');
      const writable = await fileHandle.createWritable();

      // 写入Blob数据
      console.log(`写入数据: ${wavBlob.size} 字节`);
      await writable.write(wavBlob);

      // 关闭流
      console.log('关闭文件写入流');
      await writable.close();

      console.log(`已保存调试音频到本地文件系统: ${fileName}`);
      return true;
    } catch (error) {
      console.error('保存到本地文件系统失败:', error);
      throw error; // 重新抛出错误以便上层处理
    }
  }

  // 添加一个方法，用于初始化文件系统访问
  async initFileSystemAccess() {
    try {
      // 请求用户选择目录
      this.debugDirHandle = await window.showDirectoryPicker({
        id: 'audioDebugDir',
        mode: 'readwrite',
        startIn: 'documents'
      });

      // 创建client_browser_debug子目录
      await this.debugDirHandle.getDirectoryHandle('client_browser_debug', { create: true });

      console.log('已初始化文件系统访问并创建调试目录');
      return true;
    } catch (error) {
      console.error('初始化文件系统访问失败:', error);
      return false;
    }
  }

}

export default new AudioService();