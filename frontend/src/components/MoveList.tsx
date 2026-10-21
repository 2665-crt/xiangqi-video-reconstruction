import { AnimatedDisclosure, MotionRegion } from './MotionRegion';
import { useState } from 'react';
import { CheckCircle2, ListOrdered, ScanLine } from 'lucide-react';
import type { Task } from '../types';
import { pieceText, moveTimeText, timeText } from '../types';
import { visibleRecords, reviewIntervals } from '../replay';

const labels = { auto_verified: '自动确认', corrected: '已校正', needs_review: '待确认', tentative: '推断待核对' };
const violationLabels = { illegal_move: '非法走法', multiple_pieces: '多子违规', takeback: '疑似悔棋', turn_mismatch: '轮次违规' };
export function MoveList({ task, step, onSelect, selecting = false, boundaries = [], onRecheck, onCancelSelection, disabled = false }: {
  task: Task | null; step: number; onSelect: (index: number) => void; selecting?: boolean;
  boundaries?: number[]; onRecheck?: () => void; onCancelSelection?: () => void; disabled?: boolean;
}) {
  const [view, setView] = useState<'moves' | 'reviews'>('moves');
  const records = task ? visibleRecords(task) : [];
  const intervals = task ? reviewIntervals(task) : [];
  const visible = new Set(records.map(r => r.index));
  const edits = task?.moves.filter(r => !visible.has(r.index) && ['auto_verified','corrected'].includes(r.status)
    && r.change_type !== 'review' && (r.manual_override || r.violation_note) && r.position_after) ?? [];
  return <section className={`moves-section ${view === 'reviews' ? 'show-reviews' : ''}`}><div className="panel-heading move-tabs"><ListOrdered size={19} aria-hidden="true" /><div role="group" aria-label="棋谱视图"><h2><button aria-pressed={view === 'moves'} className={view === 'moves' ? 'active' : ''} onClick={() => setView('moves')}>对局步骤</button></h2><button aria-pressed={view === 'reviews'} className={view === 'reviews' ? 'active' : ''} onClick={() => setView('reviews')}>待核对 <span className="review-count">{intervals.length}</span></button></div><span className="confirmed-count"><CheckCircle2 size={14} />{records.filter(r => r.status === 'auto_verified' || r.status === 'corrected').length} 手已确认</span></div>
    <div className="moves-tools" title="点击步骤定位动作开始前 1 秒；时间待核对的记录请查看原视频"><span className="muted">{view === 'reviews' ? `${intervals.length} 段视频待核对` : `${records.length} 手 · 点击定位原视频`}</span>{onRecheck && <button className="range-recheck-button" disabled={disabled || records.length < 2} onClick={onRecheck}><ScanLine size={14} />区域范围内重新检测</button>}</div>
    <MotionRegion show={selecting}>{selecting && <div className="range-selection" role="status"><span>{boundaries.length ? '已选一个步骤，请点击另一个步骤；顺序不限。' : '请在棋谱中点击两个步骤，选择重检边界。'}</span><button className="text-button" onClick={onCancelSelection}>取消选择</button></div>}</MotionRegion>
    {!!records.length && <p className="hint">动作开始前 1 秒 · 时间待核对的记录请查看原视频</p>}
    {!records.length || !task ? <div className="moves-empty" hidden={view !== 'moves'}><ListOrdered className="empty-lines" size={36} strokeWidth={1.4} aria-hidden="true" /><p>识别后，完整棋谱将呈现在这里</p><small>每一手都能定位回原视频</small></div> : <div className="moves-scroll" hidden={view !== 'moves'}>
      {records.map((record, index) => {
        const action = record.move ?? (record.candidates.length === 1 && ['verified', 'partial'].includes(record.board_check?.status ?? '') ? record.candidates[0] : null);
        const boundary = boundaries.includes(record.index);
        const inRange = boundaries.length === 2 && record.index > Math.min(...boundaries) && record.index < Math.max(...boundaries);
        return <button key={record.index} data-record-index={record.index} aria-current={step === record.index ? 'step' : undefined} aria-pressed={boundary} className={`move-row ${record.violation ? `violation-row violation-${record.violation}` : ''} ${step === record.index ? 'selected' : ''} ${boundary ? 'range-boundary' : inRange ? 'range-interior' : ''}`} title={record.violation_note || record.reason} onClick={() => onSelect(record.index)}>
        <span className="move-index">{String(index + 1).padStart(2, '0')}</span><span className={`side-dot ${record.move?.color ?? ''}`} />
        <span className="move-description">{action ? `${action.color === 'red' ? '红' : '黑'}${pieceText[action.color][record.position_before?.pieces.find(p => p.id === action.piece_id)?.kind ?? task.initial_position?.pieces.find(p => p.id === action.piece_id)?.kind ?? 'pawn']}  ${action.from.file + 1},${action.from.rank + 1} → ${action.to.file + 1},${action.to.rank + 1}` : `局面调整 · ${record.position_after?.pieces.length}枚棋子`}{action?.captured_piece_id && <small>吃子</small>}{record.violation && <small className="violation-description">{record.violation === 'takeback' && record.status === 'auto_verified' ? '悔棋' : violationLabels[record.violation]} · {record.violation_note || record.reason}{record.related_steps?.length ? `（涉及步骤 ${record.related_steps.join('、')}）` : ''}</small>}</span>
        <time style={{ whiteSpace: 'nowrap', flexShrink: 0, fontVariantNumeric: 'tabular-nums' }} title={record.action_start_s != null ? `动作开始 ${record.action_start_s.toFixed(3)} 秒；点击提前 1 秒观看` : '旧记录或未确定动作时间，请重新分析或核对原视频'}>{moveTimeText(record)}</time><span className={`status-pill ${record.status}`}>{labels[record.status]}</span>
      </button>; })}
    </div>}
    {view === 'reviews' && !intervals.length && <div className="moves-empty"><CheckCircle2 size={34} /><p>暂无待核对区间</p><small>识别产生的疑点会集中在这里</small></div>}
    {!!intervals.length && <AnimatedDisclosure className="review-intervals pending-intervals" open={view === 'reviews'} onToggle={open => setView(open ? 'reviews' : 'moves')} title={<> 待核对视频区间 · {intervals.length} 段（不计入走棋）</>}>
      <p className="hint">这些区间尚未确定实际走法，保留原视频证据供核对；完整回放和导出需要先解决缺口。</p>
      {intervals.map((interval, i) => <div key={interval.records[0].index} className="review-interval">
        <button className="text-button" onClick={() => onSelect(interval.records[0].index)}>核对区间 {i + 1} · {timeText(interval.start)}—{timeText(interval.end)}</button>
        <p className="hint">{interval.records.at(-1)?.reason}</p>
        {interval.records.length > 1 && <label>选择区间内证据<select aria-label={`区间${i + 1}证据`} value="" onChange={e => { if (e.target.value) onSelect(Number(e.target.value)); }}>
          <option value="">已归并 {interval.records.length} 条历史疑点</option>
          {interval.records.map(r => <option key={r.index} value={r.index}>{timeText(r.video_end_s)} · {r.reason}</option>)}
        </select></label>}
      </div>)}
    </AnimatedDisclosure>}
    {!!edits.length && <AnimatedDisclosure className="review-intervals" title={<> 已保存局面校正 · {edits.length} 条（不计走棋）</>}>
      {edits.map(r => <div key={r.index} className="review-interval"><button className="text-button" onClick={() => onSelect(r.index)}>查看局面 · {timeText(r.video_end_s)}</button><p className="hint">{r.violation_note || r.reason}</p></div>)}
    </AnimatedDisclosure>}
  </section>;
}
