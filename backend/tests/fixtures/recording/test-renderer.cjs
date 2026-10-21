const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../../../..');
const packets = [];
let active = 0, peak = 0, closed = false, failed = null;
const browser = {
  async newPage() {
    const state = { index: 0, elapsed: 0 };
    return {
      on() {}, async goto() {}, async waitForFunction() {},
      async evaluate(fn, argument) {
        if (argument && typeof argument === 'object' && 'records' in argument) return;
        if (typeof argument === 'number') {
          state.index = argument;
          state.elapsed = fn.toString().includes('1000') ? 1000 : 0;
          return 4 / 60 * 1000;
        }
        state.index = argument.index;
        state.elapsed = argument.elapsed;
      },
      locator() { return { async boundingBox() { return { x: 0, y: 0, width: 1080, height: 1192 }; } }; },
      context() {
        return { async newCDPSession() { return { async send() {
          const bytes = Buffer.from(`${state.index}:${state.elapsed.toFixed(3)}`);
          peak = Math.max(peak, ++active);
          await new Promise(resolve => setTimeout(resolve, state.elapsed === 0 ? 15 : 1));
          active--;
          return { data: bytes.toString('base64') };
        } }; } };
      },
    };
  },
  async close() { closed = true; },
};
const payload = { interval_s: 1, records: [{}] };
const context = {
  Buffer, setTimeout,
  require(name) {
    if (name === 'node:fs') return { readFileSync() { return JSON.stringify(payload); } };
    if (name.includes('playwright')) return { chromium: { async launch() { return browser; } } };
    return require(name);
  },
  process: { argv: ['node', 'renderer', 'payload', 'http://localhost'], stdout: {
    write(data) { packets.push(data); return true; },
  } },
  console: { error(message) { failed = message; } },
};
vm.runInNewContext(fs.readFileSync(path.join(root, 'scripts/render-recording.cjs'), 'utf8'), context);
(async () => {
  const deadline = Date.now() + 3000;
  while (!closed && !failed && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 5));
  assert.equal(failed, null);
  assert.equal(closed, true);
  assert.equal(peak, 2);
  const frames = [];
  for (let i = 0; i < packets.length; i += 2) {
    const header = JSON.parse(packets[i]);
    assert.equal(header.bytes, packets[i+1].length);
    frames.push([packets[i+1].toString(), header.repeat]);
  }
  assert.deepEqual(frames, [['0:0.000', 60], ['1:0.000', 1], ['1:16.667', 1],
                          ['1:33.333', 1], ['1:50.000', 1], ['1:1000.000', 56]]);
  console.log('CONCURRENT_CAPTURE_AND_ORDER_PASSED');
})().catch(error => { console.error(error); process.exitCode = 1; });
