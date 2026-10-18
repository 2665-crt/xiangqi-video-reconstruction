import { AnimatedDisclosure } from './MotionRegion';
import type { Task } from '../types';
import { pieceText, timeText } from '../types';
import { displayStep } from '../replay';

export function RecheckPanel({ task, busy, onApply, onCancel, onSelect }: {
  task: Task; busy: boolean; onApply: () => void; onCancel: () => void; onSelect: (index: number) => void;
}) {
  const job = task.recheck;
  const progress = job ? Math.max(0, Math.min(100, (job.processed_s - job.start_s) / (job.end_s - job.start_s) * 100)) : 0;
  const restored = job?.records.filter(r => r.move && ['auto_verified', 'corrected'].includes(r.status)).length ?? 0;
  return <section className="recheck-panel" aria-label="区域重检进度" aria-live="polite">
    <div className="recheck-heading"><strong>区域范围重新检测</strong><span>{job ? `${Math.floor(progress)}%` : '等待选择区间'}</span></div>
    <div className="progress-track" role="progressbar" aria-label="区域重检" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.floor(progress)}><i style={{ transform: `scaleX(${progress / 100})` }} /></div>
    {!job ? <p className="hint">点击“对局步骤”标题栏的重检按钮，再点选两个步骤。自动检测前一步落定至后一步起动之间的走棋。</p> : <>
      <p className="hint">第 {displayStep(task, job.left_index)}—{displayStep(task, job.right_index)} 手 · {job.start_s.toFixed(3)}—{job.end_s.toFixed(3)} 秒</p>
      <p className="hint">已检查至 {timeText(job.processed_s)} · {job.scanned_frames} 帧 · 已恢复 {restored} 手 · 待核对 {job.unresolved} 处</p>
      {job.integrated && <p className="hint">已自动写入对局步骤 {job.integrated_count ?? restored} 手，无需再次手动补棋。</p>}
      <p className={job.status === 'needs_review' || job.status === 'failed' ? 'rule-warning' : 'hint'}>{job.message}</p>
      {(job.integrated || ['ready', 'applied'].includes(job.status)) && <p className="hint">补回 {job.added} 手 · 纠正 {job.changed} 手 · 移除 {job.removed} 条旧记录</p>}
      {!!job.manual_conflicts.length && <p className="rule-warning">重检结果与区间内 {job.manual_conflicts.length} 处人工校正不同，原记录已留存，请对照视频核对。</p>}
      {!!job.records.length && <AnimatedDisclosure className="recheck-results" title={<> 查看恢复走法、原记录与证据</>}>
        <p className="hint">{job.integrated ? '已更新的区间步骤' : '原区间记录'}</p>
        {task.moves.slice(job.left_index, job.right_index - 1).map(r => <button key={r.index} className="text-button" onClick={() => onSelect(r.index)}>第 {displayStep(task, r.index)} 手 · {r.manual_override ? '人工校正 · ' : ''}{r.reason || '已有走法'}</button>)}
        <p className="hint">逐帧识别走法与疑点证据</p>
        {job.records.map((r, i) => {
          const m = r.move ?? (r.candidates.length === 1 ? r.candidates[0] : null);
          const piece = m && r.position_before?.pieces.find(p => p.id === m.piece_id);
          return <div className="recheck-candidate" key={i}>
            <strong>{m ? `${i + 1}. ${m.color === 'red' ? '红' : '黑'}${pieceText[m.color][piece?.kind ?? 'pawn']} ${m.from.file + 1},${m.from.rank + 1} → ${m.to.file + 1},${m.to.rank + 1}` : '待核对的视频变化'}</strong>
            <small>{r.action_start_s?.toFixed(3) ?? '起动待核对'}—{(r.action_end_s ?? r.video_end_s).toFixed(3)} 秒 · {r.reason}</small>
            {!!r.evidence_ids.length && <div className="evidence-strip">{r.evidence_ids.map((id, j) => <a key={id} href={`/api/tasks/${task.id}/evidence/${id}`} target="_blank" rel="noreferrer"><img src={`/api/tasks/${task.id}/evidence/${id}`} alt={`恢复候选${i + 1}证据`} /><small>{r.evidence_times_s?.[j]?.toFixed(3) ?? '时间待核对'} 秒</small></a>)}</div>}
          </div>;
        })}
      </AnimatedDisclosure>}
      {job.status === 'running' && <button className="text-button" disabled={busy} onClick={onCancel}>停止区域重检</button>}
      {!job.integrated && ['ready', 'needs_review'].includes(job.status) && <button className="primary" disabled={busy} onClick={onApply}>写入已识别走棋</button>}
      {job.status === 'needs_review' && <p className="hint">已确认走法自动进入对局步骤；剩余疑点仅保留为待核对视频区间，不阻止已确认步骤写入。</p>}
    </>}
  </section>;
}
