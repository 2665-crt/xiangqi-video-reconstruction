const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { randomUUID } = require('node:crypto');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const fs = require('node:fs');
const { claimInstance } = require('./instances.cjs');

const roots = () => [path.resolve('output/multi-instance-20261024', randomUUID(), 'data'), path.resolve('output/multi-instance-20261024', randomUUID(), 'videos')];

test('concurrent launches get separate persistent slots and preserve the original paths', async () => {
  const [data, videos] = roots();
  const instances = await Promise.all(Array.from({ length: 4 }, () => claimInstance(data, videos)));
  try {
    assert.deepEqual(instances.map(x => x.id).sort(), [1, 2, 3, 4]);
    const first = instances.find(x => x.id === 1);
    assert.equal(first.dataRoot, data);
    assert.equal(first.recordingRoot, videos);
    for (const instance of instances.filter(x => x.id !== 1)) {
      assert.equal(instance.dataRoot, path.join(data, 'instances', String(instance.id)));
      assert.equal(instance.recordingRoot, path.join(videos, 'instances', String(instance.id)));
    }
    const second = instances.find(x => x.id === 2);
    await second.release();
    const reopened = await claimInstance(data, videos);
    instances.push(reopened);
    assert.equal(reopened.id, 2);
    assert.equal(reopened.dataRoot, second.dataRoot);
  } finally {
    await Promise.all(instances.map(x => x.release()));
  }
});

test('different application data roots do not occupy each other slots', async () => {
  const a = await claimInstance(...roots());
  const b = await claimInstance(...roots());
  try { assert.equal(a.id, 1); assert.equal(b.id, 1); }
  finally { await a.release(); await b.release(); }
});

test('a surviving backend lock reserves its directory even before publishing any PID record', async () => {
  const [data, videos] = roots();
  const next = await claimInstance(data, videos, async candidate => candidate === data);
  try { assert.equal(next.id, 2); } finally { await next.release(); }
});

test('a stale owner PID pointing to an unrelated process does not hide historical tasks', async () => {
  const [data, videos] = roots();
  fs.mkdirSync(path.join(data, 'runtime'), { recursive: true });
  fs.writeFileSync(path.join(data, 'runtime', 'backend-owner.json'), JSON.stringify({ pid: process.pid }));
  const next = await claimInstance(data, videos);
  try { assert.equal(next.id, 1); } finally { await next.release(); }
});

test('the operating system releases a crashed instance slot', async () => {
  const [data, videos] = roots();
  const modulePath = path.join(__dirname, 'instances.cjs');
  const script = `require(${JSON.stringify(modulePath)}).claimInstance(${JSON.stringify(data)}, ${JSON.stringify(videos)}).then(x => process.stdout.write(String(x.id)))`;
  const child = spawn(process.execPath, ['-e', script], { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
  try {
    const [ready] = await once(child.stdout, 'data');
    assert.equal(String(ready), '1');
    const occupied = await claimInstance(data, videos);
    assert.equal(occupied.id, 2);
    await occupied.release();
    const exited = once(child, 'exit');
    child.kill();
    await exited;
    const recovered = await claimInstance(data, videos);
    try { assert.equal(recovered.id, 1); } finally { await recovered.release(); }
  } finally { if (child.exitCode === null) child.kill(); }
});
