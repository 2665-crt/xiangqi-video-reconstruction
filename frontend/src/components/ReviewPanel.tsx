import { AnimatedDisclosure } from './MotionRegion';
import { useEffect, useState } from 'react';
import type { Color, Move, PieceKind, Position, Square, Task } from '../types';
import { pieceText } from '../types';
import { displayStep, visibleRecords } from '../replay';

function movedPosition(position: Position, action: Move): Position {
  return { pieces: position.pieces.filter(p => p.id === action.piece_id || p.square.file !== action.to.file || p.square.rank !== action.to.rank)
    .map(p => p.id === action.piece_id ? { ...p, square: action.to } : p),
    side_to_move: action.color === 'red' ? 'black' : 'red' };
}

export function ReviewPanel({ task, index, position, onSave, onSavePosition, onClose, onPreview, picked, busy, videoTime, insert = false }: {
  task: Task; index: number; position: Position; onSave: (moves: Move[], time: number, start?: number) => void;
  onSavePosition: (position: Position, time: number, start?: number) => void; insert?: boolean;
  onClose: () => void; onPreview: (position: Position | null) => void; picked: Square | null; busy: boolean; videoTime: number;
}) {
  const record = insert ? undefined : task.moves[index - 1];
  const isInterval = !!record && !visibleRecords(task).some(r => r.index === index);
  const suggested = record?.move ?? record?.candidates[0];
  const [mode, setMode] = useState<'move' | 'position'>('move');
  const [pieceId, setPieceId] = useState(suggested?.piece_id ?? '');
  const [target, setTarget] = useState<Square | null>(suggested?.to ?? null);
  const [feedback, setFeedback] = useState('');
  const [staged, setStaged] = useState(position);
  const [time, setTime] = useState(record?.action_end_s ?? record?.video_end_s ?? videoTime);
  const [startTime, setStartTime] = useState<number | ''>(record?.action_start_s ?? (insert ? videoTime : ''));
  const [brush, setBrush] = useState('');
  const [pickingTarget, setPickingTarget] = useState(false);
  const piece = staged.pieces.find(p => p.id === pieceId);
  const action: Move | null = piece && target ? { index, piece_id: piece.id, color: piece.color, from: piece.square, to: target,
    captured_piece_id: staged.pieces.find(p => p.id !== piece.id && p.square.file === target.file && p.square.rank === target.rank)?.id ?? null } : null;
  const preview = mode === 'move' && action ? movedPosition(staged, action) : staged;

  useEffect(() => { onPreview(preview); }, [staged, pieceId, target, mode]);
  useEffect(() => {
    if (!picked || busy) return;
    setFeedback('');
    const occupant = staged.pieces.find(p => p.square.file === picked.file && p.square.rank === picked.rank);
    if (mode === 'position' && brush) {
      const pieces = staged.pieces.filter(p => p !== occupant);
      if (brush !== 'remove') {
        const [color, kind] = brush.split(':') as [Color, PieceKind];
        pieces.push({ id: occupant?.id ?? `manual_${crypto.randomUUID()}`, color, kind, square: picked });
      }
      setStaged({ ...staged, pieces }); setPieceId(''); setTarget(null);
    } else if (piece && pickingTarget) {
      if (piece.square.file === picked.file && piece.square.rank === picked.rank) {
        setPieceId(''); setTarget(null); setPickingTarget(false);
      } else if (mode === 'position') {
        setStaged(movedPosition(staged, { index, piece_id: piece.id, color: piece.color, from: piece.square, to: picked, captured_piece_id: occupant?.id ?? null }));
        setPieceId(''); setTarget(null); setPickingTarget(false);
      } else setTarget(picked);
    } else if (occupant) { setPieceId(occupant.id); setTarget(null); setPickingTarget(true); }
  }, [picked]);

  function switchMode(next: 'move' | 'position') {
    setMode(next); setPieceId(''); setTarget(null); setBrush(''); setFeedback(''); setPickingTarget(false);
    setStaged(next === 'position' ? record?.position_after ?? position : position);
  }

  return <section className="review-panel board-review"><div className="section-heading"><h3>{insert ? '插入遗漏走棋' : isInterval ? '核对视频区间' : `校正第 ${displayStep(task, index)} 手`}</h3><button className="text-button" onClick={onClose}>关闭</button></div>
    {isInterval && <p className="hint">{record.video_start_s.toFixed(3)}—{record.video_end_s.toFixed(3)} 秒 · 尚未确定实际走法，不计入对局步数。{record.reason}</p>}
    {!insert && <div className="segmented correction-modes"><button className={mode === 'move' ? 'active' : ''} onClick={() => switchMode('move')}>点选移动</button><button className={mode === 'position' ? 'active' : ''} onClick={() => switchMode('position')}>局面校正</button></div>}
    <p className="hint">{insert ? `在第 ${displayStep(task, index)} 手之前补棋，已有走法保留并顺延。先点棋子，再点落点；保存后可继续补下一手。` : mode === 'move' ? '先点棋子，再点落点。替换当前一步；起点采用前一手已保存局面。点击选中棋子可取消选择。' : '在下方选择棋子，再点棋盘添加或替换棋种；选“删除棋子”后点击删除。选择“移动”可调整位置。'}</p>
    <p className="hint">后续保留原走法，承接校正后局面。规则问题只提示，允许移动、保存和继续回放。</p>
    {record?.violation_note && <p className="rule-warning" role="status">{record.violation_note}</p>}
    {record?.board_check && <p className={record.board_check.status === 'conflict' ? 'rule-warning' : 'hint'} role="status">
      整盘核对：{record.board_check.reason}（操作前 {record.board_check.visible_before}/90 格，操作后 {record.board_check.visible_after}/90 格）
      {record.board_check.backfilled && ' · 已回查原视频补齐中间落点'}
      {!!record.board_check.before_mismatches.length && ` · 操作前差异：${record.board_check.before_mismatches.map(s => `${s.file + 1},${s.rank + 1}`).join('；')}`}
      {!!record.board_check.after_mismatches.length && ` · 操作后差异：${record.board_check.after_mismatches.map(s => `${s.file + 1},${s.rank + 1}`).join('；')}`}
    </p>}
    {feedback && <p className="alert error" role="alert">{feedback}</p>}
    {mode === 'position' && <div className="piece-palette" aria-label="局面校正棋子面板">
      <button className={!brush ? 'active' : ''} disabled={busy} onClick={() => { setBrush(''); setPieceId(''); }}>移动</button>
      {(['red', 'black'] as Color[]).flatMap(color => (Object.keys(pieceText[color]) as PieceKind[]).map(kind => <button key={`${color}:${kind}`} aria-label={`${color === 'red' ? '红' : '黑'}${pieceText[color][kind]}`} className={`${color} ${brush === `${color}:${kind}` ? 'active' : ''}`} disabled={busy} onClick={() => { setBrush(`${color}:${kind}`); setPieceId(''); }}>{pieceText[color][kind]}</button>))}
      <button className={brush === 'remove' ? 'active' : ''} disabled={busy} onClick={() => { setBrush('remove'); setPieceId(''); }}>删除棋子</button>
    </div>}
    {piece && <p className="selected-piece" role="status">已选{piece.color === 'red' ? '红' : '黑'}{pieceText[piece.color][piece.kind]}{target ? ` → 第${target.file + 1}列第${target.rank + 1}行` : '，请点击落点'} <button className="text-button" onClick={() => { setPieceId(''); setTarget(null); setPickingTarget(false); }}>重新选择</button></p>}
    {mode === 'move' && !!record?.candidates.length && <div className="candidate-buttons">{record.candidates.map((m, i) => <button key={i} disabled={busy} onClick={() => { setPieceId(m.piece_id); setTarget(m.to); }}>候选 {i + 1}</button>)}</div>}
    <label>落子时间（原视频秒）<input aria-label="落子时间" type="number" min={0} max={task.metadata.duration_s} step="any" value={time} disabled={busy} onChange={e => setTime(Number(e.target.value))} /></label>
    <label>动作开始时间（原视频秒）<input aria-label="动作开始时间" type="number" min={0} max={task.metadata.duration_s} step="any" value={startTime} disabled={busy} onChange={e => setStartTime(e.target.value === '' ? '' : Number(e.target.value))} /></label>
    {record?.action_end_s != null && record.action_end_s > task.end_s && <p className="hint">本手在选区内开始，最终落点已延伸至选区外 {record.action_end_s.toFixed(3)} 秒核对。</p>}
    <button className="primary full" aria-busy={busy} onClick={() => {
      if (busy) { setFeedback('正在提交，请等待；当前校正内容已保留。'); return; }
      if (mode === 'position') onSavePosition(staged, time, startTime === '' ? undefined : startTime);
      else if (action) onSave([action], time, startTime === '' ? undefined : startTime);
      else setFeedback('请点选移动棋子和最终落点。');
    }}>{insert ? '保存并继续补棋' : mode === 'position' ? '保存当前局面' : '确认并保存这一步'}</button>
    {record && <button className="text-button" disabled={busy} onClick={() => {
      if (window.confirm('确认此处不计入走棋？该记录将从步骤列表、计数、回放和导出中跳过。')) onSave([], record.action_end_s ?? record.video_end_s);
    }}>此处不计入走棋</button>}
    {!!record?.evidence_ids.length && <AnimatedDisclosure title={<> 查看操作前后原视频证据</>}><div className="evidence-strip">{record.evidence_ids.map((id, i) => <a key={id} href={`/api/tasks/${task.id}/evidence/${id}`} target="_blank" rel="noreferrer"><img src={`/api/tasks/${task.id}/evidence/${id}`} alt={isInterval ? '待核对视频区间证据截图' : `第${displayStep(task, index)}手原视频证据截图`} /><small>{record.evidence_times_s?.[i] != null ? `${record.evidence_times_s[i]!.toFixed(3)} 秒` : '历史截图，时间未记录'}</small></a>)}</div></AnimatedDisclosure>}
  </section>;
}
