import type { Task } from '../types';

export function RecoveryPanel({ task, busy, stopping, onResume, onStop }: {
  task: Task; busy: boolean; stopping: boolean; onResume: () => void; onStop: () => void;
}) {
  const waiting = task.status === 'waiting_service';
  if (!waiting && !['failed', 'interrupted', 'action_required'].includes(task.status)) return null;
  const recovery = task.recovery;
  const remaining = recovery ? Math.max(0, Math.ceil(recovery.next_attempt_at - Date.now() / 1000)) : 0;
  return <div className={`recovery-panel ${waiting ? '' : 'recovery-attention'}`} role="status" aria-live="polite">
    <strong>{waiting ? '任务已保存，后台将自动恢复' : task.status === 'action_required' ? '任务已保存，需要管理员检查服务' : '已有棋谱已保留，可以恢复任务'}</strong>
    <p>{waiting ? `已保存 ${task.moves.length} 条记录。${remaining > 0 ? `约 ${remaining} 秒后再次尝试` : '等待后台继续处理'}，无需点击重连。` : recovery ? '恢复会保留已有记录；修改设置或棋局前请先停止自动恢复。' : '旧任务将创建恢复副本并重新处理，原任务及棋谱保留。'}</p>
    {recovery?.last_error && <small>{recovery.phase === 'response' ? '模型服务或网关响应未完成' : recovery.phase === 'upload' ? '截图发送未完成' : '服务连接未完成'} · 已尝试 {recovery.attempts} 次</small>}
    <div className="recovery-actions">
      <button className="secondary" disabled={busy || stopping} onClick={onResume}>{busy ? '正在提交…' : waiting ? '立即尝试恢复' : '检查并恢复任务'}</button>
      {recovery?.auto_run && <button className="text-button" disabled={busy || stopping} onClick={onStop}>{stopping ? '正在停止…' : '停止自动恢复'}</button>}
    </div>
  </div>;
}
