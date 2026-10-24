import type { Calibration, Move, Position, RecordingJob, ResourceUsage, Task, TaskSummary } from './types';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === 'string' ? data.detail : '请求失败，请检查输入并重试');
  }
  return response.json();
}
const json = (method: string, body: unknown): RequestInit => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
export const api = {
  resources: (signal?: AbortSignal) => request<ResourceUsage>('/api/resources', signal ? { signal } : undefined),
  health: () => request<{ status: string; vision_configured: boolean; detail?: string }>('/api/health'),
  list: () => request<TaskSummary[]>('/api/tasks'),
  get: (id: string) => request<Task>(`/api/tasks/${id}`),
  exportRecording: (task: Task, interval_s: number) => request<RecordingJob>(`/api/tasks/${task.id}/recordings`, json('POST', { revision: task.revision, interval_s })),
  recording: (id: string) => request<RecordingJob>(`/api/recordings/${id}`),
  latestRecording: (taskId: string) => request<RecordingJob | null>(`/api/tasks/${taskId}/recordings/latest`),
  upload: (file: File) => { const body = new FormData(); body.append('file', file); return request<Task>('/api/tasks', { method: 'POST', body }); },
  setup: (task: Task, start_s: number, end_s: number, calibration: Calibration | null, initial_position?: Position) =>
    request<Task>(`/api/tasks/${task.id}/setup`, json('PATCH', { revision: task.revision, start_s, end_s, calibration, initial_position })),
  analyze: (id: string) => request<Task>(`/api/tasks/${id}/analyze`, { method: 'POST' }),
  resume: (task: Task) => request<Task>(`/api/tasks/${task.id}/resume`, json('POST', { revision: task.revision })),
  stop: (id: string) => request<Task>(`/api/tasks/${id}/stop`, { method: 'POST' }),
  recognizeInitial: (task: Task) => request<Task>(`/api/tasks/${task.id}/recognize-initial`, json('POST', { revision: task.revision })),
  correct: (task: Task, index: number, moves: Move[], video_time_s: number, action_start_s?: number) => request<Task>(`/api/tasks/${task.id}/moves/${index}`, json('PATCH', { revision: task.revision, moves, video_time_s, action_start_s })),
  correctPosition: (task: Task, index: number, position: Position, video_time_s: number, action_start_s?: number) => request<Task>(`/api/tasks/${task.id}/moves/${index}`, json('PATCH', { revision: task.revision, position, video_time_s, action_start_s })),
  insert: (task: Task, index: number, move: Move, video_time_s: number, action_start_s?: number) => request<Task>(`/api/tasks/${task.id}/moves/${index}/insert`, json('POST', { revision: task.revision, move, video_time_s, action_start_s })),
  recheck: (task: Task, first_index: number, second_index: number) => request<Task>(`/api/tasks/${task.id}/recheck`, json('POST', { revision: task.revision, first_index, second_index })),
  applyRecheck: (task: Task) => request<Task>(`/api/tasks/${task.id}/recheck/apply`, json('POST', { revision: task.revision, recheck_id: task.recheck?.id })),
  cancelRecheck: (task: Task) => request<Task>(`/api/tasks/${task.id}/recheck/cancel`, json('POST', { revision: task.revision, recheck_id: task.recheck?.id })),
};
