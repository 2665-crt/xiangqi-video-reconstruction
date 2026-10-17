import type { Task } from '../types';
import { pieceText, timeText } from '../types';
import { visibleRecords } from '../replay';

const labels = { auto_verified: '自动确认', corrected: '已校正', needs_review: '待确认', tentative: '待复核' };
const violationLabels = { illegal_move: '非法走法', multiple_pieces: '多子违规', takeback: '疑似悔棋', turn_mismatch: '轮次违规' };
export function MoveList({ task, step, onSelect }: { task: Task | null; step: number; onSelect: (index: number) => void }) {
  const records = task ? visibleRecords(task) : [];
  return <section className="moves-section"><div className="section-heading"><h3>对局步骤</h3><span className="muted">{records.length} 手</span></div>
    {!!records.length && <p className="hint">原视频时间 · 点击步骤定位到对应画面</p>}
    {!records.length || !task ? <div className="moves-empty"><span className="empty-lines">☷</span><p>识别后，完整棋谱将呈现在这里</p><small>每一手都能定位回原视频</small></div> : <div className="moves-scroll">
      {records.map((record, index) => <button key={record.index} className={`move-row ${record.violation ? `violation-row violation-${record.violation}` : ''} ${step === record.index ? 'selected' : ''}`} title={record.violation_note || record.reason} onClick={() => onSelect(record.index)}>
        <span className="move-index">{String(index + 1).padStart(2, '0')}</span><span className={`side-dot ${record.move?.color ?? ''}`} />
        <span className="move-description">{record.move ? `${record.move.color === 'red' ? '红' : '黑'}${pieceText[record.move.color][record.position_after?.pieces.find(p => p.id === record.move?.piece_id)?.kind ?? task.initial_position?.pieces.find(p => p.id === record.move?.piece_id)?.kind ?? 'pawn']}  ${record.move.from.file + 1},${record.move.from.rank + 1} → ${record.move.to.file + 1},${record.move.to.rank + 1}` : record.change_type === 'review' && record.status === 'corrected' ? '人工审核 · 此处不计入走棋' : record.position_after && ['auto_verified', 'corrected'].includes(record.status) ? `局面调整 · ${record.position_after.pieces.length}枚棋子` : '需要核对这处画面'}{record.move?.captured_piece_id && <small>吃子</small>}{record.violation && <small className="violation-description">{violationLabels[record.violation]} · {record.violation_note || record.reason}{record.related_steps?.length ? `（涉及步骤 ${record.related_steps.join('、')}）` : ''}</small>}</span>
        <time style={{ whiteSpace: 'nowrap', flexShrink: 0, fontVariantNumeric: 'tabular-nums' }} dateTime={`PT${Math.floor(record.video_end_s)}S`} title={`原视频时间 ${timeText(record.video_end_s)}`}>{timeText(record.video_end_s)}</time><span className={`status-pill ${record.status}`}>{labels[record.status]}</span>
      </button>)}
    </div>}
  </section>;
}
