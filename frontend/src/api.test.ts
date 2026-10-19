import { afterEach, expect, test, vi } from 'vitest';
import { api } from './api';
import type { Task } from './types';

afterEach(() => vi.unstubAllGlobals());

test('resume preserves the saved operation instead of starting a new analysis', async () => {
  expect(api).toHaveProperty('resume');
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'waiting_service' }) });
  vi.stubGlobal('fetch', fetch);
  const resume = (api as unknown as { resume: (task: Task) => Promise<Task> }).resume;
  const result = await resume({ id: 'task-1', revision: 8 } as Task);
  expect(result.status).toBe('waiting_service');
  const [url, options] = fetch.mock.calls[0];
  expect(url).toBe('/api/tasks/task-1/resume');
  expect(JSON.parse(options.body)).toEqual({ revision: 8 });
});

test('stop submits cancellation without starting another analysis', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'interrupted' }) });
  vi.stubGlobal('fetch', fetch);
  await api.stop('task-1');
  expect(fetch).toHaveBeenCalledWith('/api/tasks/task-1/stop', { method: 'POST' });
  expect(fetch).toHaveBeenCalledTimes(1);
});

test('initial recognition submits the saved revision without starting full analysis', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ initial_candidates: [] }) });
  vi.stubGlobal('fetch', fetch);
  await api.recognizeInitial({ id: 'task-1', revision: 7 } as Task);
  expect(fetch).toHaveBeenCalledTimes(1);
  const [url, options] = fetch.mock.calls[0];
  expect(url).toBe('/api/tasks/task-1/recognize-initial');
  expect(options.method).toBe('POST');
  expect(JSON.parse(options.body)).toEqual({ revision: 7 });
});

test('recording export submits the reviewed revision and compact pace', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ id: 'job', status: 'queued' }) });
  vi.stubGlobal('fetch', fetch);
  const job = await api.exportRecording({ id: 'task-1', revision: 9 } as Task, 1);
  expect(job.status).toBe('queued');
  const [url, options] = fetch.mock.calls[0];
  expect(url).toBe('/api/tasks/task-1/recordings');
  expect(options.method).toBe('POST');
  expect(JSON.parse(options.body)).toEqual({ revision: 9, interval_s: 1 });
});
