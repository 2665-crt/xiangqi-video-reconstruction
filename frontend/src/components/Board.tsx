import { useLayoutEffect, useRef, useState } from 'react';
import type { Calibration, InitialPieceReview, Move, Piece, Position, Square } from '../types';
import { pieceText } from '../types';

export function planBoardTransition(before: Position | null, after: Position | null, stepDelta: number, replace = false, orientationChanged = false) {
  const old = new Map(before?.pieces.map(p => [p.id, p]) ?? []);
  const next = new Map(after?.pieces.map(p => [p.id, p]) ?? []);
  const removed = [...old.values()].filter(p => !next.has(p.id));
  const added = [...next.values()].filter(p => !old.has(p.id));
  const moved: { before: Piece; after: Piece }[] = [];
  let identityChanged = false;
  for (const p of next.values()) {
    const previous = old.get(p.id);
    if (!previous) continue;
    if (previous.color !== p.color || previous.kind !== p.kind) identityChanged = true;
    if (previous.square.file !== p.square.file || previous.square.rank !== p.square.rank) moved.push({ before: previous, after: p });
  }
  let type: 'none' | 'move' | 'replace' = 'replace';
  if (!moved.length && !removed.length && !added.length && !identityChanged && !orientationChanged) type = 'none';
  else if (before && after && !replace && !orientationChanged && !identityChanged && Math.abs(stepDelta) <= 1 && moved.length === 1 && removed.length <= 1 && added.length <= 1) {
    const move = moved[0];
    const capturedAtTarget = removed.every(p => p.square.file === move.after.square.file && p.square.rank === move.after.square.rank);
    const restoredAtSource = added.every(p => p.square.file === move.before.square.file && p.square.rank === move.before.square.rank);
    if (capturedAtTarget && restoredAtSource && !(removed.length && added.length)) type = 'move';
  }
  return { type, moved, removed, added };
}

interface Props { position: Position | null; orientation?: Calibration['orientation']; step?: number; replace?: boolean; lastMove?: Move | null; inferredMove?: Move | null; onSquare?: (square: Square) => void; selected?: Square | null; reviews?: InitialPieceReview[] }
export function Board({ position, orientation = 'normal', step = 0, replace = false, lastMove, inferredMove, onSquare, selected, reviews = [] }: Props) {
  const previous = useRef<{ position: Position | null; step: number; orientation: Calibration['orientation'] }>({ position: null, step, orientation });
  const nodes = useRef(new Map<string, SVGGElement>());
  const piecesLayer = useRef<SVGGElement>(null);
  const ghostsLayer = useRef<SVGGElement>(null);
  const markersLayer = useRef<SVGGElement>(null);
  const animations = useRef<Animation[]>([]);
  const [marker, setMarker] = useState<{ from: Square; to: Square; color: string; pieceId: string } | null>(null);
  const transition = planBoardTransition(previous.current.position, position, step - previous.current.step, replace, previous.current.orientation !== orientation);
  const activeId = transition.moved[0]?.after.id ?? marker?.pieceId;
  const rotated = orientation === 'rotated_180' || orientation === 'mirrored_rotated_180';
  const mirrored = orientation === 'mirrored' || orientation === 'mirrored_rotated_180';
  const gridX = (file: number) => 50 + file * 47;
  const gridY = (rank: number) => 50 + rank * 47;
  const x = (file: number) => gridX(rotated !== mirrored ? 8 - file : file);
  const y = (rank: number) => gridY(rotated ? 9 - rank : rank);

  useLayoutEffect(() => {
    const before = previous.current.position;
    const oldOrientation = previous.current.orientation;
    const orientationChanged = oldOrientation !== orientation;
    const plan = planBoardTransition(before, position, step - previous.current.step, replace, orientationChanged);
    previous.current = { position, step, orientation };
    if (plan.type === 'none') return;
    animations.current.forEach(animation => animation.cancel());
    animations.current = [];
    ghostsLayer.current?.replaceChildren();
    const motion = plan.type === 'move' ? plan.moved[0] : null;
    setMarker(motion ? { from: motion.before.square, to: motion.after.square, color: motion.after.color, pieceId: motion.after.id }
      : lastMove ? { from: lastMove.from, to: lastMove.to, color: lastMove.color, pieceId: lastMove.piece_id } : null);
    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      // 临时保留旧棋子的画面，让吃子在落子时消失，跳转只过渡棋子层。
      for (const piece of plan.type === 'replace' ? before?.pieces ?? [] : plan.removed) {
        const node = nodes.current.get(piece.id);
        if (!node || !ghostsLayer.current) continue;
        const ghost = node.cloneNode(true) as SVGGElement;
        ghost.removeAttribute('data-piece-id');
        const oldRotated = oldOrientation === 'rotated_180' || oldOrientation === 'mirrored_rotated_180';
        const oldMirrored = oldOrientation === 'mirrored' || oldOrientation === 'mirrored_rotated_180';
        ghost.setAttribute('transform', `translate(${gridX(oldRotated !== oldMirrored ? 8 - piece.square.file : piece.square.file)} ${gridY(oldRotated ? 9 - piece.square.rank : piece.square.rank)})`);
        ghostsLayer.current.appendChild(ghost);
        const animation = ghost.animate([{ opacity: 1 }, { opacity: 0 }], { duration: plan.type === 'move' ? 140 : 180, delay: plan.type === 'move' ? 370 : 0, fill: 'both' });
        animation.onfinish = () => ghost.remove();
        animations.current.push(animation);
      }
      if (motion) {
        const node = nodes.current.get(motion.after.id);
        if (node) {
          const from = `translate(${x(motion.before.square.file)}px, ${y(motion.before.square.rank)}px)`;
          const to = `translate(${x(motion.after.square.file)}px, ${y(motion.after.square.rank)}px)`;
          animations.current.push(node.animate([{ transform: from }, { transform: to, offset: .8 }, { transform: to }], { duration: 520, easing: 'cubic-bezier(.25,.65,.3,1)' }));
          const face = node.querySelector('.piece-face')!;
          animations.current.push(face.animate([
            { transform: 'translateY(0px) scale(1)' },
            { transform: 'translateY(-6px) scale(1.28)', offset: .2 },
            { transform: 'translateY(-6px) scale(1.28)', offset: .65 },
            { transform: 'translateY(0px) scale(1)', offset: 1 },
          ], { duration: 520, easing: 'ease-in-out' }));
          animations.current.push(node.querySelector('.piece-ground-shadow')!.animate([
            { opacity: .18, transform: 'scale(1)' }, { opacity: .3, transform: 'scale(1.3)', offset: .3 }, { opacity: .18, transform: 'scale(1)' },
          ], { duration: 520 }));
        }
        for (const piece of plan.added) {
          const node = nodes.current.get(piece.id);
          if (node) animations.current.push(node.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 140, delay: 370, fill: 'backwards' }));
        }
        if (markersLayer.current) animations.current.push(markersLayer.current.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 160, delay: 360, fill: 'backwards' }));
      } else if (piecesLayer.current) {
        animations.current.push(piecesLayer.current.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 180 }));
      }
    }
    for (const id of nodes.current.keys()) if (!position?.pieces.some(p => p.id === id)) nodes.current.delete(id);
  }, [position, step, replace, lastMove, orientation]);

  useLayoutEffect(() => () => {
    animations.current.forEach(animation => animation.cancel());
    ghostsLayer.current?.replaceChildren();
    previous.current = { ...previous.current, position: null, step: 0 };
  }, []);

  const orderedPieces = [...position?.pieces ?? []].sort((a, b) => Number(a.id === activeId) - Number(b.id === activeId));
  return <div className="board-frame"><svg viewBox="0 0 476 525" className="board-svg" role="group" aria-label="虚拟象棋棋盘">
    <defs><linearGradient id="wood" x2="1" y2="1"><stop stopColor="#f3e4cb" /><stop offset=".55" stopColor="#e9d2ad" /><stop offset="1" stopColor="#dfc197" /></linearGradient>
      <radialGradient id="piece"><stop stopColor="#fff9e9" /><stop offset=".85" stopColor="#f5e4bf" /><stop offset="1" stopColor="#dfbc85" /></radialGradient>
      <filter id="shadow" x="-30%" y="-30%" width="160%" height="180%"><feDropShadow dx="0" dy="2" stdDeviation="2" floodOpacity=".2" /></filter>
    </defs>
    <rect width="476" height="525" rx="7" fill="url(#wood)" />
    <rect x="38" y="38" width="400" height="447" fill="none" stroke="#886440" strokeWidth="2" />
    {Array.from({ length: 10 }, (_, rank) => <line key={`h${rank}`} x1="50" x2="426" y1={gridY(rank)} y2={gridY(rank)} stroke="#967650" strokeWidth="1.2" />)}
    {Array.from({ length: 9 }, (_, file) => <g key={`v${file}`} stroke="#967650" strokeWidth="1.2"><line x1={gridX(file)} x2={gridX(file)} y1="50" y2={file === 0 || file === 8 ? 473 : gridY(4)} />{file > 0 && file < 8 && <line x1={gridX(file)} x2={gridX(file)} y1={gridY(5)} y2="473" />}</g>)}
    {[0, 7].map(rank => <g key={rank} stroke="#967650"><line x1={gridX(3)} y1={gridY(rank)} x2={gridX(5)} y2={gridY(rank + 2)} /><line x1={gridX(5)} y1={gridY(rank)} x2={gridX(3)} y2={gridY(rank + 2)} /></g>)}
    <text x="138" y="271" textAnchor="middle" className="river-text">楚 河</text><text x="338" y="271" textAnchor="middle" className="river-text">汉 界</text>
    {Array.from({ length: 9 }, (_, file) => <text key={file} x={x(file)} y="509" textAnchor="middle" className="board-number">{file + 1}</text>)}
    {Array.from({ length: 10 }, (_, rank) => <text key={rank} x="18" y={y(rank) + 4} textAnchor="middle" className="board-number">{rank + 1}</text>)}
    <g ref={markersLayer} className="move-markers" aria-hidden="true">{marker && <>
      <circle cx={x(marker.from.file)} cy={y(marker.from.rank)} r="5" fill="#fffef6" stroke="#8c8d78" strokeWidth="1" />
      <circle cx={x(marker.from.file)} cy={y(marker.from.rank)} r="8" fill="none" stroke="#fffef6" strokeWidth="1.2" />
      <circle className={`landing-glow ${marker.color}`} cx={x(marker.to.file)} cy={y(marker.to.rank)} r="23" fill="none" stroke={marker.color === 'red' ? '#eb6652' : '#72c762'} strokeWidth="2.5" />
    </>}</g>
    <g ref={ghostsLayer} className="piece-ghosts" aria-hidden="true" />
    <g ref={piecesLayer} className="pieces-layer">{orderedPieces.map(p => <g key={p.id} ref={node => { if (node) nodes.current.set(p.id, node); }} data-piece-id={p.id} transform={`translate(${x(p.square.file)} ${y(p.square.rank)})`} className="chess-piece">
      <ellipse className="piece-ground-shadow" cy="4" rx="20" ry="17" fill="#302517" opacity=".18" />
      <g className="piece-face"><circle r="20" fill="url(#piece)" stroke="#b89360" strokeWidth="1.3" filter="url(#shadow)" /><circle r="16.5" fill="none" stroke={p.color === 'red' ? '#a94734' : '#343b37'} strokeWidth=".8" />
        <text y="7.8" textAnchor="middle" fill={p.color === 'red' ? '#a94734' : '#29312d'} className="piece-text">{pieceText[p.color][p.kind]}</text>
      </g>
    </g>)}</g>
    {inferredMove && <g className="inferred-move-marker" role="img" aria-label="推断落点，待核对">
      <circle cx={x(inferredMove.to.file)} cy={y(inferredMove.to.rank)} r="25" fill="none" stroke="#a86e1d" strokeWidth="3" strokeDasharray="5 3" />
      <title>推断落点，待核对</title>
    </g>}
    {reviews.map(r => <g key={`review-${r.square.file}-${r.square.rank}`} className="piece-review-marker" aria-label={`待核对：第${r.square.file + 1}列第${r.square.rank + 1}行`}>
      {r.state === 'occupancy' ? <rect x={x(r.square.file) - 22} y={y(r.square.rank) - 22} width="44" height="44" fill="#c077161a" stroke="#c07716" strokeWidth="2" strokeDasharray="5 3" /> : <circle cx={x(r.square.file)} cy={y(r.square.rank)} r="24" fill="none" stroke="#c07716" strokeWidth="3" strokeDasharray="5 3" />}
      {r.prediction === null && <text x={x(r.square.file)} y={y(r.square.rank) + 8} textAnchor="middle" fill="#9b5b0b" fontSize="26">{r.state === 'occupancy' ? '·' : '?'}</text>}
      <circle cx={x(r.square.file) + 17} cy={y(r.square.rank) - 18} r="8" fill="#c07716" /><text x={x(r.square.file) + 17} y={y(r.square.rank) - 14} textAnchor="middle" fill="white" fontSize="12">!</text>
    </g>)}
    {selected && <circle cx={x(selected.file)} cy={y(selected.rank)} r="24" fill="none" stroke="#227661" strokeWidth="3" />}
    {onSquare && Array.from({ length: 90 }, (_, i) => <circle key={i} cx={x(i % 9)} cy={y(Math.floor(i / 9))} r="22" fill="transparent" role="button" tabIndex={0}
      aria-label={`棋盘第${i % 9 + 1}列第${Math.floor(i / 9) + 1}行`} onClick={() => onSquare({ file: i % 9, rank: Math.floor(i / 9) })}
      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSquare({ file: i % 9, rank: Math.floor(i / 9) }); } }} />)}
    {!position && <g><rect x="103" y="202" width="270" height="96" rx="8" fill="#fffaf0" fillOpacity=".94" /><text x="238" y="238" textAnchor="middle" className="board-empty-title">等待识别棋局</text><text x="238" y="270" textAnchor="middle" className="board-empty-sub">上传视频后，逐步复现每一手</text></g>}
  </svg></div>;
}
