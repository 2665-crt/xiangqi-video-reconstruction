import { useEffect, useState } from 'react';
import { api } from '../api';
import type { RecordingJob, Task } from '../types';
import { visibleRecords } from '../replay';

export function recordingBlockReason(task: Task): string {
  if (!task.initial_position) return '请先确认并保存初始棋局';
  return visibleRecords(task).length || task.moves.some(r => r.change_type === 'setup' && r.manual_override && r.position_after)
    ? '' : '暂无可导出的对局步骤';
}

export function RecordingPanel({ task, intervalS }: { task: Task; intervalS: number }) {
  const [job, setJob] = useState<RecordingJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const running = job?.status === 'queued' || job?.status === 'running';
  const reason = recordingBlockReason(task);
  useEffect(() => {
    let disposed = false;
    api.latestRecording(task.id).then(value => { if (!disposed) setJob(value); })
      .catch(error => { if (!disposed) setError(error.message); })
      .finally(() => { if (!disposed) setLoading(false); });
    return () => { disposed = true; };
  }, [task.id]);

  useEffect(() => {
    if (!job || !running) return;
    let disposed = false;
    const timer = window.setInterval(() => api.recording(job.id).then(value => {
      if (!disposed) { setJob(value); setError(''); }
    }).catch(error => { if (!disposed) setError(`读取导出进度失败：${error.message}`); }), 1000);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [job?.id, running]);

  async function exportVideo() {
    setSubmitting(true); setError('');
    try { setJob(await api.exportRecording(task, intervalS)); }
    catch (error) { setError(error instanceof Error ? error.message : '导出未完成，请重试'); }
    finally { setSubmitting(false); }
  }

  return <section className="recording-panel" aria-label="对局录像导出">
    <div className="recording-heading"><div><strong>对局录像</strong><p>仅录棋盘 · MP4 · 60 帧／秒 · {intervalS} 秒／手</p></div>
      <button className="primary" aria-busy={submitting || running} disabled={!!reason || loading || submitting || running} onClick={() => void exportVideo()}>{submitting || running ? '正在生成录像…' : '导出对局录像'}</button>
    </div>
    {reason ? <p className="recording-warning">{reason}</p> : <p className="hint">按当前已保存步骤导出，未校正内容不会阻止导出。</p>}
    {error && <p className="recording-warning" role="alert">{error}</p>}
    {job && <div className="recording-result" role="status">
      <div className="recording-status"><span>{job.stage}</span><span>{job.progress}%</span></div>
      <div className="progress-track" role="progressbar" aria-label="录像生成进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={job.progress}><i style={{ transform: `scaleX(${job.progress / 100})` }} /></div>
      {job.error && <p className="recording-warning">{job.error}</p>}
      {job.status === 'completed' && <>
        <p>{job.duration_s} 秒 · {job.width} × {job.height} · {job.interval_s} 秒／手</p>
        <p className="recording-path">{job.file_path}</p>
        <a href={`/api/recordings/${job.id}/video`} target="_blank" rel="noreferrer">查看录像</a>
        {job.revision !== task.revision && <p className="recording-warning">此录像来自之前的棋局版本；修改棋局后请重新导出。</p>}
      </>}
    </div>}
    <p className="recording-directory">保存到：F:\codex产出\虚拟棋盘对战录制\录屏存放</p>
  </section>;
}
