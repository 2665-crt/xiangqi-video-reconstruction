const { BrowserWindow } = require('electron');
const fs = require('node:fs');

async function writeFrame(png, repeat) {
  await new Promise((resolve, reject) => process.stdout.write(
    Buffer.concat([Buffer.from(JSON.stringify({ bytes: png.length, repeat }) + '\n'), png]),
    error => error ? reject(error) : resolve(),
  ));
}

exports.renderRecording = async (manifestPath, url) => {
  const parsed = new URL(url);
  if (parsed.protocol !== 'http:' || parsed.hostname !== '127.0.0.1' || parsed.pathname !== '/recording-render') {
    throw new Error('无效的本地录像页面');
  }
  const payload = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const windows = [];
  const errors = [];
  try {
    const renderers = await Promise.all([0, 1].map(async () => {
      const window = new BrowserWindow({ width: 1080, height: 1192, useContentSize: true, show: false,
        webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true, offscreen: true, backgroundThrottling: false, partition: 'recording' } });
      windows.push(window);
      window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
      window.webContents.on('will-navigate', (event, next) => { if (new URL(next).origin !== parsed.origin) event.preventDefault(); });
      window.webContents.on('console-message', details => { if (details.level === 'error') errors.push(details.message); });
      await window.loadURL(url);
      await window.webContents.executeJavaScript(`window.loadRecording(${JSON.stringify(payload)})`);
      const bounds = await window.webContents.executeJavaScript(`(()=>{const box=document.querySelector('.board-svg').getBoundingClientRect();return{x:Math.round(box.x),y:Math.round(box.y),width:Math.round(box.width),height:Math.round(box.height)}})()`);
      return { window, async captureFrame() {
        const image = await new Promise(resolve => {
          window.webContents.once('paint', (_event, _dirty, bitmap) => resolve(bitmap));
          window.webContents.invalidate();
        });
        return image.crop(bounds).resize({ width: 1080, height: 1192 }).toPNG();
      } };
    }));
    await writeFrame(await renderers[0].captureFrame(), payload.interval_s * 60);
    for (let index = 1; index <= payload.records.length; index++) {
      const durations = await Promise.all(renderers.map(({ window }) => window.webContents.executeJavaScript(`window.renderRecordingStep(${index},0)`)));
      const movingFrames = Math.ceil(durations[0] / 1000 * 60);
      for (let frame = 0; frame < movingFrames; frame += renderers.length) {
        const images = await Promise.all(renderers.slice(0, movingFrames - frame).map(async ({ window, captureFrame }, offset) => {
          await window.webContents.executeJavaScript(`window.renderRecordingStep(${index},${(frame + offset) * 1000 / 60})`);
          return captureFrame();
        }));
        for (const image of images) await writeFrame(image, 1);
      }
      await Promise.all(renderers.map(({ window }) => window.webContents.executeJavaScript(`window.renderRecordingStep(${index},1000)`)));
      await writeFrame(await renderers[0].captureFrame(), payload.interval_s * 60 - movingFrames);
    }
    if (errors.length) throw new Error(errors.join('\n'));
  } finally {
    for (const window of windows) if (!window.isDestroyed()) window.destroy();
  }
};
