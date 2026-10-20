const { app, BrowserWindow, Menu, dialog, session } = require('electron');
const { spawn, spawnSync } = require('node:child_process');
const { randomUUID } = require('node:crypto');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');

const productName = 'XiangQi Vedio Reconstruction';
const renderIndex = process.argv.indexOf('--recording-render');
const portableRoot = app.isPackaged ? path.dirname(process.execPath) : path.resolve(__dirname, '..');
const dataRoot = process.env.XIANGQI_DATA_ROOT || path.join(portableRoot, 'data');
const recordingRoot = process.env.XIANGQI_RECORDING_ROOT || path.join(portableRoot, '录屏存放');
app.setName(productName);
app.setAppUserModelId('com.2665crt.xiangqi-reconstruction');
const profileRoot = renderIndex >= 0
  ? path.join(path.dirname(process.argv[renderIndex + 1]), 'renderer-profile')
  : path.join(dataRoot, 'desktop');
fs.mkdirSync(profileRoot, { recursive: true });
app.setPath('userData', profileRoot);

if (renderIndex >= 0) {
  app.disableHardwareAcceleration();
  app.commandLine.appendSwitch('force-device-scale-factor', '1');
  app.whenReady().then(() => require('./recording.cjs').renderRecording(
    process.argv[renderIndex + 1], process.argv[renderIndex + 2],
  )).then(() => app.quit()).catch(error => {
    process.stderr.write(String(error.stack || error) + '\n');
    app.exit(1);
  });
} else if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  let window;
  let backend;
  let origin;
  let spawnError;
  let quitting = false;
  const token = randomUUID();
  const readyFile = path.join(dataRoot, 'runtime', randomUUID() + '.json');

  function log(message) {
    const logRoot = path.join(dataRoot, 'logs');
    fs.mkdirSync(logRoot, { recursive: true });
    fs.appendFileSync(path.join(logRoot, 'desktop.log'), new Date().toISOString() + ' ' + message + '\n');
  }

  function localRequest(url, method = 'GET', headers = {}) {
    return new Promise((resolve, reject) => {
      const request = http.request(url, { method, headers, timeout: 2500 }, response => {
        const chunks = [];
        response.on('data', chunk => chunks.push(chunk));
        response.on('end', () => {
          try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
          catch (error) { reject(error); }
        });
      });
      request.on('timeout', () => request.destroy(new Error('本地服务尚未就绪')));
      request.on('error', reject);
      request.end();
    });
  }

  async function stopBackend() {
    if (!backend || backend.exitCode !== null) return;
    if (origin) {
      try { await localRequest(origin + '/api/desktop/shutdown', 'POST', { 'X-Desktop-Token': token }); }
      catch (error) { log('Shutdown request: ' + error.message); }
    }
    const deadline = Date.now() + 15000;
    while (backend.exitCode === null && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 100));
    if (backend.exitCode === null) {
      spawnSync(path.join(process.env.SystemRoot, 'System32', 'taskkill.exe'), ['/PID', String(backend.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
    }
    fs.rmSync(readyFile, { force: true });
  }

  function secureWindow(target) {
    target.webContents.setWindowOpenHandler(({ url }) => {
      const destination = new URL(url);
      if (origin && destination.origin === origin && /^\/api\/(recordings\/[^/]+\/video|tasks\/[^/]+\/evidence\/[^/]+)$/.test(destination.pathname)) {
        return { action: 'allow', overrideBrowserWindowOptions: {
          title: productName, width: 1080, height: 900, autoHideMenuBar: true,
          webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true },
        } };
      }
      return { action: 'deny' };
    });
    target.webContents.on('will-navigate', (event, url) => {
      if (!origin || new URL(url).origin !== origin) event.preventDefault();
    });
    target.webContents.on('page-title-updated', event => { event.preventDefault(); target.setTitle(productName); });
    target.webContents.on('did-create-window', child => secureWindow(child));
  }

  app.on('second-instance', () => {
    if (window) { if (window.isMinimized()) window.restore(); window.focus(); }
  });
  app.on('window-all-closed', () => app.quit());
  app.on('before-quit', event => {
    if (!quitting) {
      event.preventDefault();
      quitting = true;
      stopBackend().finally(() => app.quit());
    }
  });

  app.whenReady().then(async () => {
    fs.mkdirSync(dataRoot, { recursive: true });
    fs.mkdirSync(recordingRoot, { recursive: true });
    Menu.setApplicationMenu(null);
    session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
    window = new BrowserWindow({
      title: productName, width: 1500, height: 1000, minWidth: 1000, minHeight: 720,
      backgroundColor: '#f4f5ee', autoHideMenuBar: true,
      webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true },
    });
    secureWindow(window);
    window.on('closed', () => app.quit());
    await window.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(`<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>body{margin:0;background:#f4f5ee;color:#28594c;font-family:Segoe UI,Microsoft YaHei,sans-serif;display:grid;place-items:center;height:100vh}.card{text-align:center;animation:fade .3s ease}h1{font-size:25px;font-weight:500}p{font-size:14px;color:#748575}.mark{width:52px;height:52px;display:grid;place-items:center;border:2px solid #789d83;border-radius:12px;margin:auto;font-size:30px}.line{width:160px;height:3px;background:#dce4d6;overflow:hidden;margin:26px auto}.line:after{content:'';display:block;width:55px;height:100%;background:#789d83;animation:slide 1.4s ease-in-out infinite}@keyframes slide{from{transform:translateX(-55px)}to{transform:translateX(160px)}}@keyframes fade{from{opacity:0}to{opacity:1}}@media(prefers-reduced-motion:reduce){*{animation:none!important}}</style><div class="card"><div class="mark">棋</div><h1>${productName}</h1><p>正在启动本地模型，请稍候…</p><div class="line"></div><p>所有识别在本机完成</p></div></html>`));
    const resources = app.isPackaged ? process.resourcesPath : process.env.XIANGQI_DESKTOP_RESOURCES;
    const backendExe = path.join(resources, 'backend', 'XiangQiBackend.exe');
    const environment = { ...process.env,
      XIANGQI_DESKTOP_TOKEN: token, XIANGQI_DESKTOP_EXE: process.execPath,
      YOLO_OFFLINE: 'true', YOLO_AUTOINSTALL: 'false',
      PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', PYTHONNOUSERSITE: '1',
    };
    delete environment.PYTHONHOME;
    delete environment.PYTHONPATH;
    delete environment.ELECTRON_RUN_AS_NODE;
    backend = spawn(backendExe, [
      '--data-root', dataRoot, '--recording-root', recordingRoot,
      '--frontend-root', path.join(resources, 'frontend'), '--ready-file', readyFile,
    ], { cwd: path.dirname(backendExe), env: environment, windowsHide: true, stdio: 'ignore' });
    backend.on('error', error => { spawnError = error; log('Backend spawn: ' + error.message); });
    const deadline = Date.now() + 120000;
    let detail = '正在加载模型';
    while (Date.now() < deadline) {
      if (quitting) return;
      if (spawnError) throw spawnError;
      if (backend.exitCode !== null) throw new Error('后台服务已退出，请查看 data/logs/backend.log。');
      if (fs.existsSync(readyFile)) {
        try {
          origin = JSON.parse(fs.readFileSync(readyFile, 'utf8')).url;
          const health = await localRequest(origin + '/api/health');
          if (health.status === 'ok' && health.vision_configured) break;
          detail = health.detail || '本地模型未就绪';
        } catch (error) { detail = error.message; }
      }
      await new Promise(resolve => setTimeout(resolve, 250));
    }
    if (!origin || Date.now() >= deadline) throw new Error('启动超时：' + detail);
    log('Backend ready at ' + origin);
    await window.loadURL(origin);
    window.setTitle(productName);
    backend.on('exit', code => {
      if (!quitting) { dialog.showErrorBox(productName, '后台服务已退出（' + code + '）。请查看 data/logs/backend.log 后重新打开应用。'); app.quit(); }
    });
  }).catch(error => {
    if (!quitting) {
      try { log(String(error.stack || error)); } catch {}
      dialog.showErrorBox(productName, '启动未完成：' + error.message + '\n请将完整便携文件夹解压到可写目录，再重新打开。');
      app.quit();
    }
  });
}
