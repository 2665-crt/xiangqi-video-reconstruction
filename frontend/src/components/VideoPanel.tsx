import { useRef, useState, type RefObject } from 'react';
import { Upload } from 'lucide-react';
import type { Calibration, Task } from '../types';
import { timeText } from '../types';
import { gridPoint } from '../calibration';

interface Props {
  task: Task | null; videoRef: RefObject<HTMLVideoElement | null>; editing: boolean;
  calibration: Calibration | null; setCalibration: (value: Calibration) => void;
  onTime: (time: number, seeking: boolean) => void; onUpload: (file: File) => void; busy: boolean;
  startTime: number;
}
export function VideoPanel({ task, videoRef, editing, calibration, setCalibration, onTime, onUpload, busy, startTime }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<number | null>(null);
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);
  function updatePoint(clientX: number, clientY: number) {
    if (!task || !calibration || drag.current === null || !svg.current) return;
    const box = svg.current.getBoundingClientRect();
    const point = { x: Math.max(0, Math.min(task.metadata.width - 1, (clientX - box.left) / box.width * task.metadata.width)),
      y: Math.max(0, Math.min(task.metadata.height - 1, (clientY - box.top) / box.height * task.metadata.height)) };
    setCalibration({ ...calibration, corners: calibration.corners.map((p, i) => i === drag.current ? point : p) });
  }
  return <>
    <input ref={input} type="file" accept="video/mp4,.mp4" className="hidden-input" aria-label="上传录屏视频" disabled={busy} onChange={e => { const file = e.target.files?.[0]; if (file) onUpload(file); e.target.value = ''; }} />
    {!task ? <button className={`upload-zone ${dragging && !busy ? 'dragging' : ''}`} disabled={busy} aria-busy={busy} onClick={() => input.current?.click()}
      onDragOver={e => { e.preventDefault(); e.dataTransfer.dropEffect = busy ? 'none' : 'copy'; }}
      onDragEnter={e => { e.preventDefault(); dragDepth.current++; if (!busy) setDragging(true); }}
      onDragLeave={() => { dragDepth.current = Math.max(0, dragDepth.current - 1); if (!dragDepth.current) setDragging(false); }}
      onDrop={e => { e.preventDefault(); dragDepth.current = 0; setDragging(false); const file = e.dataTransfer.files[0]; if (file && !busy) onUpload(file); }}>
      <span className="upload-symbol"><Upload size={31} strokeWidth={1.5} aria-hidden="true" /></span>
      <strong>{busy ? '正在上传与检查视频…' : '上传你的象棋录屏'}</strong><span>拖入视频，或点击选择文件</span><small>MP4 · 固定视角 · 棋盘完整可见 · 最大 5 GiB</small>
    </button> : <>
      <div className="video-stage">
        {editing ? <><img src={`/api/tasks/${task.id}/frame?time_s=${startTime}`} alt="用于校准棋盘的起始时间画面" />
          {calibration && <svg ref={svg} className="calibration-layer" viewBox={`0 0 ${task.metadata.width} ${task.metadata.height}`} onPointerMove={e => updatePoint(e.clientX, e.clientY)} onPointerUp={() => { drag.current = null; }}>
            <polygon points={calibration.corners.map(p => `${p.x},${p.y}`).join(' ')} fill="#4cd5a51c" stroke="#62e9b6" strokeWidth={task.metadata.width / 260} />
            {Array.from({ length: 19 }, (_, i) => {
              const a = i < 9 ? gridPoint(calibration.corners, i / 8, 0) : gridPoint(calibration.corners, 0, (i - 9) / 9);
              const b = i < 9 ? gridPoint(calibration.corners, i / 8, 1) : gridPoint(calibration.corners, 1, (i - 9) / 9);
              return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#62e9b6" strokeWidth={task.metadata.width / 650} pointerEvents="none" />;
            })}
            {calibration.corners.map((p, i) => <g key={i} onPointerDown={e => { e.preventDefault(); drag.current = i; e.currentTarget.setPointerCapture(e.pointerId); }} style={{ cursor: 'grab' }}>
              <circle cx={p.x} cy={p.y} r={task.metadata.width / 50} fill="#62e9b6" stroke="#123b31" strokeWidth={task.metadata.width / 350} />
              <text x={p.x} y={p.y + task.metadata.width / 140} textAnchor="middle" fontSize={task.metadata.width / 42} fill="#14352b">{i + 1}</text>
            </g>)}
          </svg>}</> : <video key={task.id} ref={videoRef} src={`/api/tasks/${task.id}/video`} controls preload="metadata" onTimeUpdate={e => onTime(e.currentTarget.currentTime, e.currentTarget.seeking)} onSeeking={e => onTime(e.currentTarget.currentTime, true)} />}
      </div>
      <div className="video-meta"><span className="file-name" title={task.filename}>{task.filename}</span><span>{task.metadata.width} × {task.metadata.height} · {timeText(task.metadata.duration_s)}</span></div>
      {editing && <p className="hint">拖动 1—4 号角点，依次对准棋盘左上、右上、右下、左下的最外侧交叉点。</p>}
      <button className="text-button" disabled={busy} onClick={() => input.current?.click()}>更换录屏视频</button>
    </>}
  </>;
}
