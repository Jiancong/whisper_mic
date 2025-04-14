const { exec } = require('child_process');
const path = require('path');

// 启动开发服务器
const startDev = () => {
  console.log('启动Vue开发服务器...');
  const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  
  const devProcess = exec(`${npm} run dev`, {
    cwd: __dirname
  });
  
  devProcess.stdout.on('data', (data) => {
    console.log(data.toString());
  });
  
  devProcess.stderr.on('data', (data) => {
    console.error(data.toString());
  });
  
  devProcess.on('close', (code) => {
    console.log(`开发服务器已退出，退出码: ${code}`);
  });
};

// 启动应用
startDev();