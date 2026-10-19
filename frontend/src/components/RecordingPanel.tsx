import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Clapperboard, LoaderCircle } from 'lucide-react';
import { api } from '../api';
import type { RecordingJob, Task } from '../types';
import { visibleRecords } from '../replay';

export function recordingBlockReason(task: Task): string {
  if (!task.initial_position) return '请先确认并保存初始棋局';
  return visibleRecords(task).length || task.moves.some(r => r.change_type === 'setup' && r.manual_override && r.position_after)
    ? '' : '暂无可导出的对局步骤';
}

export function RecordingFeedback({ job, revision, error }: { job: RecordingJob | null; revision: number; error: string }) {
  const failure = error || job?.error || (job?.status === 'failed' ? '录像生成失败，请重试' : '');
  return <div className="recording-feedback">
    {failure && <span className="recording-warning" role="alert">{failure}</span>}
    {job?.status === 'completed' && <>
      <a href={`/api/recordings/${job.id}/video`} target="_blank" rel="noreferrer">{job.revision !== revision ? '查看旧版本录像' : '查看录像'}</a>
      {job.revision !== revision && <span className="recording-warning">棋局已修改，请重新导出。</span>}
    </>}
  </div>;
}

export function RecordingPanel({ task, intervalS, toolbarTarget }: { task: Task; intervalS: number; toolbarTarget?: HTMLElement | null }) {
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

  const exportButton = <button className="primary export-button" title={reason || '按当前已保存步骤生成棋盘录像'} aria-busy={submitting || running} disabled={!!reason || loading || submitting || running} onClick={() => void exportVideo()}>
    {submitting || running ? <LoaderCircle size={18} className="spin" /> : <Clapperboard size={18} />}<span>{submitting || running ? `正在生成 ${job?.progress ?? 0}%` : '导出对局录像'}</span>
  </button>;
  return <section className="recording-panel" aria-label="对局录像导出">
    {toolbarTarget && createPortal(<>{exportButton}<div className="export-feedback"><RecordingFeedback job={job} revision={task.revision} error={error} /></div></>, toolbarTarget)}
    <div className="recording-heading"><div><strong>对局录像</strong><p>仅录棋盘 · MP4 · 60 帧／秒 · {intervalS} 秒／手</p></div>
      {!toolbarTarget && exportButton}
    </div>
    {reason ? <p className="recording-warning">{reason}</p> : <p className="hint">按当前已保存步骤导出，未校正内容不会阻止导出。</p>}
    <RecordingFeedback job={job} revision={task.revision} error={error} />
    {job && <div className="recording-result" role="status">
      <div className="recording-status"><span>{job.stage}</span><span>{job.progress}%</span></div>
      <div className="progress-track" role="progressbar" aria-label="录像生成进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={job.progress}><i style={{ transform: `scaleX(${job.progress / 100})` }} /></div>
      {job.status === 'completed' && <>
        <p>{job.duration_s} 秒 · {job.width} × {job.height} · {job.interval_s} 秒／手</p>
        <p className="recording-path">{job.file_path}</p>
      </>}
    </div>}
    <p className="recording-directory">保存到：项目目录下的“录屏存放”文件夹</p>
  </section>;
}
