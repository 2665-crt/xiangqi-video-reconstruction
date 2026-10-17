const fs = require('node:fs');
const { once } = require('node:events');
const { chromium } = require('../frontend/node_modules/playwright');

async function writeFrame(png, repeat) {
  if (!process.stdout.write(JSON.stringify({ bytes: png.length, repeat }) + '\n')) await once(process.stdout, 'drain');
  if (!process.stdout.write(png)) await once(process.stdout, 'drain');
}

(async () => {
  const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const errors = [];
    const renderers = await Promise.all([0, 1].map(async () => {
      const page = await browser.newPage({ viewport: { width: 1080, height: 1192 }, deviceScaleFactor: 1, reducedMotion: 'no-preference' });
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(process.argv[3], { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForFunction(() => typeof window.loadRecording === 'function');
      await page.evaluate(payload => window.loadRecording(payload), payload);
      const bounds = await page.locator('.board-svg').boundingBox();
      if (!bounds) throw new Error('录制棋盘未显示');
      const session = await page.context().newCDPSession(page);
      const clip = { ...bounds, scale: 1 };
      return { page, async captureFrame() {
        const { data } = await session.send('Page.captureScreenshot', {
          format: 'png', clip, fromSurface: true, captureBeyondViewport: false, optimizeForSpeed: true,
        });
        return Buffer.from(data, 'base64');
      } };
    }));
    await writeFrame(await renderers[0].captureFrame(), payload.interval_s * 60);
    for (let index = 1; index <= payload.records.length; index++) {
      const durations = await Promise.all(renderers.map(({ page }) => page.evaluate(index => window.renderRecordingStep(index, 0), index)));
      const movingFrames = Math.ceil(durations[0] / 1000 * 60);
      for (let frame = 0; frame < movingFrames; frame += renderers.length) {
        const images = await Promise.all(renderers.slice(0, movingFrames - frame).map(async ({ page, captureFrame }, offset) => {
          await page.evaluate(({ index, elapsed }) => window.renderRecordingStep(index, elapsed), { index, elapsed: (frame + offset) * 1000 / 60 });
          return captureFrame();
        }));
        for (const image of images) await writeFrame(image, 1);
      }
      await Promise.all(renderers.map(({ page }) => page.evaluate(index => window.renderRecordingStep(index, 1000), index)));
      await writeFrame(await renderers[0].captureFrame(), payload.interval_s * 60 - movingFrames);
    }
    if (errors.length) throw new Error(errors.join('\n'));
  } finally { await browser.close(); }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
