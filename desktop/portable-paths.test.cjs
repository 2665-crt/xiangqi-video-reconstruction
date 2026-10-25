const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { spawnSync } = require('node:child_process');
const { randomUUID } = require('node:crypto');

function startupPaths(environment, packaged = true) {
  const source = fs.readFileSync(path.join(__dirname, 'main.cjs'), 'utf8');
  const executable = path.resolve('output', 'portable-test', 'temporary-runtime', 'application.exe');
  let paths;
  const electron = { app: { isPackaged: packaged, setName() {}, setAppUserModelId() {} } };
  vm.runInNewContext(source, {
    require(name) {
      if (name === 'electron') return electron;
      if (name === './instances.cjs') return { claimInstance(dataRoot, recordingRoot) {
        paths = { dataRoot, recordingRoot };
        return new Promise(() => {});
      } };
      return require(name);
    },
    process: { argv: [executable], execPath: executable, resourcesPath: path.resolve('output', 'resources'),
      env: { XIANGQI_DESKTOP_RESOURCES: path.resolve('output', 'resources'), ...environment } },
    __dirname, __filename: path.join(__dirname, 'main.cjs'),
  });
  return { ...paths, executable };
}

test('single-file portable data stays beside the downloaded EXE, outside temporary extraction', () => {
  const portable = path.resolve('output', '中文 空格目录');
  const saved = startupPaths({ PORTABLE_EXECUTABLE_DIR: portable });
  assert.equal(saved.dataRoot, path.join(portable, 'data'));
  assert.equal(saved.recordingRoot, path.join(portable, '录屏存放'));
});

test('ordinary ZIP application still uses the full application directory', () => {
  const saved = startupPaths({});
  assert.equal(saved.dataRoot, path.join(path.dirname(saved.executable), 'data'));
});

test('explicit task and recording roots override the portable location', () => {
  const data = path.resolve('output', 'separate-data');
  const recordings = path.resolve('output', 'separate-recordings');
  const saved = startupPaths({ PORTABLE_EXECUTABLE_DIR: path.resolve('output', 'portable'),
    XIANGQI_DATA_ROOT: data, XIANGQI_RECORDING_ROOT: recordings });
  assert.equal(saved.dataRoot, data);
  assert.equal(saved.recordingRoot, recordings);
});

test('source-mode launch ignores unrelated portable launcher environment', () => {
  const saved = startupPaths({ PORTABLE_EXECUTABLE_DIR: path.resolve('output', 'unrelated') }, false);
  assert.equal(saved.dataRoot, path.resolve(__dirname, '..', 'data'));
});

test('portable data-lock retry relaunches the outer portable EXE', async () => {
  const launcher = path.resolve('output', 'portable application.exe');
  let relaunched;
  const app = { isPackaged: true, setName() {}, setAppUserModelId() {}, setPath() {}, on() {},
    whenReady: () => Promise.resolve(), relaunch: options => { relaunched = options; }, quit() {} };
  class BrowserWindow {
    webContents = { setWindowOpenHandler() {}, on() {} };
    on() {}
    async loadURL() {}
  }
  const fakeFs = { mkdirSync() {}, appendFileSync() {}, existsSync: () => false };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'main.cjs'), 'utf8'), {
    require(name) {
      if (name === 'electron') return { app, BrowserWindow, Menu: { setApplicationMenu() {} },
        session: { defaultSession: { setPermissionRequestHandler() {} } }, dialog: { showErrorBox() {} } };
      if (name === './instances.cjs') return { claimInstance: async (dataRoot, recordingRoot) => ({ id: 1, dataRoot, recordingRoot }) };
      if (name === 'node:fs') return fakeFs;
      if (name === 'node:child_process') return { spawn: () => ({ pid: 1, exitCode: 73, on() {} }) };
      return require(name);
    },
    process: { argv: ['application.exe'], execPath: path.resolve('output', 'temporary-runtime', 'application.exe'),
      resourcesPath: path.resolve('output', 'resources'), env: { PORTABLE_EXECUTABLE_FILE: launcher } },
    __dirname, Buffer, setTimeout,
  });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(relaunched?.execPath, launcher);
});

test('portable preparation screen builds under Windows PowerShell without a user profile', () => {
  const target = path.resolve('output', 'portable-startup-20261025', `splash-${randomUUID()}.bmp`);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const rendered = spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
    '-File', path.join(__dirname, 'portable-splash.ps1'), '-OutputFile', target], { windowsHide: true, encoding: 'utf8' });
  assert.equal(rendered.status, 0, rendered.stderr);
  const bitmap = fs.readFileSync(target);
  assert.equal(bitmap.subarray(0, 2).toString(), 'BM');
  assert.equal(bitmap.readInt32LE(18), 640);
  assert.equal(bitmap.readInt32LE(22), 220);
  assert.equal(bitmap.readInt16LE(28), 24);
});
