import { useEffect, useRef, useState } from 'react';
import { Check, CheckCircle2, ChevronLeft, ChevronRight, Clapperboard, FolderOpen, Grid2X2, Info, Pause, Play, Settings2, SkipBack, Video } from 'lucide-react';
import { api } from './api';
import { Board } from './components/Board';
import { VideoPanel } from './components/VideoPanel';
import { MoveList } from './components/MoveList';
import { ReviewPanel } from './components/ReviewPanel';
import { RecordingPanel } from './components/RecordingPanel';
import { RecoveryPanel } from './components/RecoveryPanel';
import { RecheckPanel } from './components/RecheckPanel';
import { ResourceMonitor } from './components/ResourceMonitor';
import { AnimatedDisclosure, MotionRegion } from './components/MotionRegion';
import { displayStep, isIgnored, lastKnownStep, nextStep, previousStep, positionAt, positionBefore, stepAtVideoTime, stepSeekTime, visibleRecords, replayRecords, reviewIntervals, recheckRange } from './replay';
import type { Calibration, Color, Piece, PieceKind, Position, Square, Task, TaskSummary } from './types';
import { pieceText, timeText, isAutomaticRecord, isPredictedRecord } from './types';

function opening(): Position {
  const pieces: Piece[] = [];
  const back: PieceKind[] = ['rook', 'horse', 'elephant', 'advisor', 'king', 'advisor', 'elephant', 'horse', 'rook'];
  for (const color of ['red', 'black'] as Color[]) {
    back.forEach((kind, file) => pieces.push({ id: `${color}_${kind}_${file}`, color, kind, square: { file, rank: color === 'red' ? 9 : 0 } }));
    [1, 7].forEach(file => pieces.push({ id: `${color}_cannon_${file}`, color, kind: 'cannon', square: { file, rank: color === 'red' ? 7 : 2 } }));
    [0, 2, 4, 6, 8].forEach(file => pieces.push({ id: `${color}_pawn_${file}`, color, kind: 'pawn', square: { file, rank: color === 'red' ? 6 : 3 } }));
  }
  return { pieces, side_to_move: 'red' };
}

export default function App() {
  const [exportTarget, setExportTarget] = useState<HTMLDivElement | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [task, setTask] = useState<Task | null>(null);
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [configured, setConfigured] = useState(false);
  const [configurationError, setConfigurationError] = useState('');
  const [connected, setConnected] = useState(false);
  const [clock, setClock] = useState(Date.now());
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [boardJump, setBoardJump] = useState(false);
  const [interval, setIntervalSeconds] = useState(2);
  const [mode, setMode] = useState<'演绎' | '对照'>('演绎');
  const [editingCalibration, setEditingCalibration] = useState(false);
  const [calibration, setCalibration] = useState<Calibration | null>(null);
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(0);
  const [draft, setDraft] = useState<Position | null>(null);
  const [candidateIndex, setCandidateIndex] = useState(0);
  const [brush, setBrush] = useState<string>('red:pawn');
  const [reviewIndex, setReviewIndex] = useState<number | null>(null);
  const [picked, setPicked] = useState<Square | null>(null);
  const [inserting, setInserting] = useState(false);
  const [selectingRange, setSelectingRange] = useState(false);
  const [rangeSteps, setRangeSteps] = useState<number[]>([]);
  const [reviewDraft, setReviewDraft] = useState<Position | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const replayTask = useRef<Task | null>(null);
  const submitting = useRef(false);
  const calibrationBeforeEdit = useRef<Calibration | null>(null);
  const initialLoaded = useRef('');
  const followAnalysis = useRef(true);
  const locked = busy || task?.status === 'analyzing' || !!task?.recovery?.auto_run || task?.recheck?.status === 'running';
  useEffect(() => { setReviewDraft(null); }, [reviewIndex, task?.id]);
  useEffect(() => {
    if (task?.recheck?.integrated) {
      setRangeSteps([task.recheck.left_index, task.recheck.right_index]);
      setStep(task.recheck.left_index); setReviewIndex(null); setReviewDraft(null); setPicked(null); setInserting(false);
    }
  }, [task?.recheck?.id, task?.recheck?.integrated]);
  useEffect(() => { const timer = window.setInterval(() => setClock(Date.now()), 1000); return () => window.clearInterval(timer); }, []);

  useEffect(() => {
    let disposed = false;
    const check = () => api.health().then(h => { if (!disposed) { setConfigured(h.vision_configured); setConfigurationError(h.detail ?? ''); setConnected(true); } }).catch(() => { if (!disposed) setConnected(false); });
    void check();
    const healthTimer = window.setInterval(check, 3000);
    api.list().then(setTasks).catch(() => {});
    const id = localStorage.getItem('xiangqi-task-id');
    if (id) api.get(id).then(updated => { if (!disposed) setTask(current => current && (current.id !== updated.id || current.revision > updated.revision) ? current : updated); })
      .catch(() => { if (!disposed) setError('读取进度失败，服务恢复后将自动同步。'); });
    const restoreTimer = window.setInterval(() => {
      const saved = localStorage.getItem('xiangqi-task-id');
      if (saved) api.get(saved).then(updated => { if (!disposed) setTask(current => current && current.id !== updated.id ? current : current && current.revision > updated.revision ? current : updated); }).catch(() => {});
    }, 3000);
    return () => { disposed = true; window.clearInterval(healthTimer); window.clearInterval(restoreTimer); };
  }, []);

  useEffect(() => {
    followAnalysis.current = true;
    setPlaying(false); setStep(0); setDraft(null); setReviewIndex(null); setPicked(null); setEditingCalibration(false); setInserting(false); setSelectingRange(false); setRangeSteps([]);
    setSettingsOpen(!!task && !task.initial_position);
    if (!task) return;
    localStorage.setItem('xiangqi-task-id', task.id);
    setStart(task.start_s); setEnd(task.end_s);
    setCalibration(task.calibration ?? { corners: [
      { x: task.metadata.width * .08, y: task.metadata.height * .08 }, { x: task.metadata.width * .92, y: task.metadata.height * .08 },
      { x: task.metadata.width * .92, y: task.metadata.height * .92 }, { x: task.metadata.width * .08, y: task.metadata.height * .92 },
    ], orientation: 'normal' });
  }, [task?.id]);

  useEffect(() => {
    if (task?.initial_draft && !task.initial_position && task.status !== 'analyzing') {
      const key = `${task.id}:${task.initial_frame_s}:${JSON.stringify(task.initial_draft)}:${JSON.stringify(task.initial_piece_reviews)}`;
      if (initialLoaded.current === key || editingCalibration) return;
      initialLoaded.current = key;
      setDraft(structuredClone(task.initial_draft));
      setStep(0); setReviewIndex(null);
    }
  }, [task?.id, task?.initial_draft, task?.status, editingCalibration]);

  useEffect(() => {
    if (!task) return;
    let disposed = false;
    const handle = window.setInterval(() => api.get(task.id).then(updated => {
      if (disposed) return;
      setTask(current => !current || current.id !== updated.id || current.revision > updated.revision ? current : updated);
      setError(current => current.startsWith('读取进度失败') ? '' : current);
      if (!reviewIndex) setStep(current => current > updated.moves.length ? lastKnownStep(updated) : current);
      if (mode === '对照' && video.current && !draft && !reviewIndex) {
        setStep(stepAtVideoTime(updated, video.current.currentTime));
      } else if (followAnalysis.current && !reviewIndex && !playing && lastKnownStep(updated) > lastKnownStep(task)) {
        setStep(lastKnownStep(updated)); setReviewIndex(null);
      }
    }).catch(() => { if (!disposed) setError('读取进度失败，服务恢复后将自动同步。'); }), 2000);
    return () => { disposed = true; window.clearInterval(handle); };
  }, [task?.id, task?.status, task?.confirmed_prefix_length, task?.moves.length, reviewIndex, mode, playing, draft]);

  useEffect(() => { replayTask.current = task; }, [task]);
  useEffect(() => {
    if (!playing || !task) return;
    const handle = window.setInterval(() => setStep(current => {
      const activeTask = replayTask.current;
      if (!activeTask) return current;
      const next = nextStep(activeTask, current);
      if (next === null) { setPlaying(false); return current; }
      return next;
    }), interval * 1000);
    return () => window.clearInterval(handle);
  }, [playing, interval, task?.id]);

  async function perform(action: () => Promise<void>) {
    if (submitting.current) { setNotice('正在提交本次操作，请等待结果。'); return; }
    submitting.current = true;
    setError(''); setNotice(''); setBusy(true);
    try { await action(); setTasks(await api.list()); }
    catch (e) { setNotice(''); setError(e instanceof Error ? e.message : '操作未完成，请重试'); }
    finally { submitting.current = false; setBusy(false); }
  }

  function selectStep(index: number) {
    if (!task) return;
    if (selectingRange) {
      if (locked) return;
      if (rangeSteps[0] === index) { setNotice('请选择另一个步骤作为边界。'); return; }
      if (!rangeSteps.length) { setRangeSteps([index]); return; }
      try {
        const selected = recheckRange(task, rangeSteps[0], index);
        setRangeSteps([selected.left, selected.right]); setSelectingRange(false);
        void perform(async () => {
          const latest = await api.get(task.id);
          setTask(await api.recheck(latest, selected.left, selected.right));
        });
      } catch (e) { setError(e instanceof Error ? e.message : '请选择有效边界'); }
      return;
    }
    followAnalysis.current = false;
    setInserting(false);
    setPlaying(false); setBoardJump(false); setDraft(null); setPicked(null);
    if (!positionAt(task, index)) {
      let previous = Math.min(index - 1, lastKnownStep(task));
      while (previous > 0 && !positionAt(task, previous)) previous--;
      setStep(previous);
      setReviewIndex(index);
    } else { setStep(index); setReviewIndex(null); }
    const time = index === 0 ? task.start_s : task.moves[index - 1] ? stepSeekTime(task.moves[index - 1]) : undefined;
    if (video.current && time !== undefined) { video.current.pause(); video.current.currentTime = time; }
  }

  function chooseRange() {
    if (!task || locked) return;
    setPlaying(false); video.current?.pause(); setReviewIndex(null); setReviewDraft(null); setPicked(null);
    setInserting(false); setSelectingRange(true); setRangeSteps([]); setError(''); setNotice('');
  }

  function editInitial() {
    if (!task) return;
    setPlaying(false); setStep(0); setReviewIndex(null);
    setDraft(structuredClone(task.initial_position ?? task.initial_candidates[0] ?? task.initial_draft ?? opening()));
  }

  function paint(square: Square) {
    if (!draft) { setPicked(square); return; }
    const pieces = draft.pieces.filter(p => p.square.file !== square.file || p.square.rank !== square.rank);
    if (brush !== 'remove') {
      const [color, kind] = brush.split(':') as [Color, PieceKind];
      pieces.push({ id: `${color}_${kind}_${square.file}_${square.rank}`, color, kind, square });
    }
    setDraft({ ...draft, pieces });
  }

  function saveSettings(action: 'save' | 'initial' | 'analyze' | 'confirm' = 'save') {
    if (!task || locked) return;
    const changed = start !== task.start_s || end !== task.end_s || JSON.stringify(calibration) !== JSON.stringify(task.calibration);
    if (task.moves.length && changed && !window.confirm('更改分析范围或棋盘定位会重新生成棋谱。确认保存这些设置？')) return;
    if (action === 'analyze' && task.moves.length && !window.confirm('重新分析将替换当前棋谱。确认重新分析？')) return;
    if (action === 'confirm' && task.initial_piece_reviews?.some(r => r.state === 'occupancy' || r.prediction === null)
      && !window.confirm('当前还有占位或棋种未知的位置。保存将以右侧手动棋盘为准，没有放置棋子的未知位置将确认为空。请核对原画面，确认保存？')) return;
    void perform(async () => {
      const saved = await api.setup(task, start, end, calibration, action === 'confirm' ? draft ?? undefined : undefined);
      setCalibration(saved.calibration);
      setTask(saved); setDraft(null); setEditingCalibration(false); setStep(0); setReviewIndex(null);
      if (action === 'analyze') { setTask(await api.analyze(saved.id)); }
      else if (action === 'initial') {
        setNotice('正在核对相邻起始画面及棋子局部图…');
        const recognized = await api.recognizeInitial(saved);
        setTask(recognized); setCandidateIndex(0);
        const candidate = recognized.initial_candidates[0] ?? recognized.initial_draft;
        if (candidate) {
          setDraft(structuredClone(candidate));
          initialLoaded.current = `${recognized.id}:${recognized.initial_frame_s}:${JSON.stringify(recognized.initial_draft)}:${JSON.stringify(recognized.initial_piece_reviews)}`;
          setNotice(recognized.initial_piece_reviews?.length ? '已复刻可识别部分；问号表示棋种未知，方框表示占位不确定。可继续解析或手动核对。' : '起始画面已识别。请核对右侧棋子的位置、数量和下一方，再确认保存。');
        } else setNotice('没有得到可用局面，请换一张清晰的起始画面，或手动设置棋局。');
      } else setNotice(action === 'confirm' ? '初始棋局已确认并保存，可以开始分析视频。' : '设置已保存，可继续识别或手动复盘。');
    });
  }

  async function stopAnalysis() {
    if (!task || stopping) return;
    setStopping(true); setPlaying(false); video.current?.pause(); setError('');
    try {
      const saved = await api.stop(task.id);
      setTask(saved); setReviewIndex(null);
      setNotice('分析已停止，已有棋谱和进度已保留。已发出的供应方请求可能仍会计费。');
      setTasks(await api.list());
    } catch (e) { setError(e instanceof Error ? e.message : '停止未完成，请重试'); }
    finally { setStopping(false); }
  }

  function resumeAnalysis() {
    if (!task) return;
    void perform(async () => {
      const latest = await api.get(task.id);
      const saved = await api.resume(latest);
      setTask(saved);
      setNotice(saved.id === task.id ? '恢复任务已排队，后台会自动继续处理。' : '恢复副本已创建，原任务和已有棋谱已保留。');
    });
  }

  const display = draft ?? reviewDraft ?? (task && reviewIndex && task.rules_enabled ? positionBefore(task, reviewIndex) : null) ?? (task?.replay_mode === 'actual' && !task.rules_enabled && reviewIndex ? task.moves[reviewIndex - 1]?.position_after : null) ?? (task ? positionAt(task, step) : null) ?? (step === 0 ? task?.initial_draft ?? null : null);
  const initialReviews = draft || (!task?.initial_position && step === 0) ? task?.initial_piece_reviews ?? [] : [];
  const reviewPosition = task && reviewIndex ? positionBefore(task, reviewIndex) : null;
  const records = task ? visibleRecords(task) : [];
  const intervals = task ? reviewIntervals(task) : [];
  const confirmedTime = records.filter(r => isAutomaticRecord(r) || r.status === 'corrected').reduce((t, r) => Math.max(t, r.action_end_s ?? r.video_end_s), task?.start_s ?? 0);
  const total = records.length;
  const confirmedRecords = records.filter(r => isAutomaticRecord(r) || r.status === 'corrected').length;
  const confirmedMoves = confirmedRecords;
  const lastKnown = task ? lastKnownStep(task) : 0;
  const progress = task ? Math.max(0, Math.min(100, (task.processed_video_s - task.start_s) / (task.end_s - task.start_s) * 100)) : 0;
  const scanTime = task ? Math.max(task.scan_progress_s ?? task.start_s, task.processed_video_s) : 0;
  const scanProgress = task ? Math.max(0, Math.min(100, (scanTime - task.start_s) / (task.end_s - task.start_s) * 100)) : 0;
  const requestSeconds = task?.recovery?.request_started_at ? Math.max(0, Math.floor(clock / 1000 - task.recovery.request_started_at)) : 0;
  const lastMove = task?.moves[step - 1]?.move;
  const settingsChanged = !!task && (start !== task.start_s || end !== task.end_s || JSON.stringify(calibration) !== JSON.stringify(task.calibration));
  const tokenEntries = task?.usage.filter(entry => typeof entry.total_tokens === 'number') ?? [];
  const recordedTokens = tokenEntries.reduce((sum, entry) => sum + Number(entry.total_tokens), 0);

  return <div className="app-shell studio">
    <header className="topbar">
      <div className="brand"><span className="brand-mark">棋</span><div><strong>棋局复刻</strong><small>XIANGQI REPLAY STUDIO</small></div></div>
      <div className="studio-project"><h1>象棋对局复盘</h1><span className={`save-state ${connected ? 'online' : ''}`} role="status"><CheckCircle2 size={15} />{busy ? '处理中' : settingsChanged || draft ? '未保存更改' : reviewIndex ? '校正待确认' : connected ? task ? '已保存' : '本地工作台' : '服务未连接'}</span></div>
      <div className="header-actions"><label className="task-picker"><FolderOpen size={18} /><select aria-label="打开已保存任务" title={task?.filename ?? '打开已保存任务'} value={task?.id ?? ''} onChange={e => { const id = e.target.value; if (id) void perform(async () => { setTask(await api.get(id)); }); }} disabled={locked}><option value="">打开已保存任务</option>{tasks.map(t => <option key={t.id} value={t.id}>{t.filename} · {t.steps} 手</option>)}</select></label><div className="studio-export" ref={setExportTarget}>{!task && <button className="primary export-button" disabled><Clapperboard size={18} />导出对局录像</button>}</div><ResourceMonitor /></div>
    </header>
    <main>
      <nav className="studio-workflow" aria-label="复盘流程">{['准备视频', '识别', '核对', '导出'].map((label, index) => {
        const phase = task?.status === 'analyzing' ? 1 : task?.initial_position ? 2 : 0;
        return <div key={label} className={`workflow-step ${index === phase ? 'current' : index < phase ? 'complete' : ''}`} aria-current={index === phase ? 'step' : undefined}><span className="workflow-number">{index < phase ? <Check size={15} /> : index + 1}</span><span>{label}</span></div>;
      })}</nav>
      <MotionRegion show={!!(error || task?.error)}><div className="alert error" role="alert">{error || task?.error}</div></MotionRegion>
      {task && <RecoveryPanel task={task} busy={busy || !connected} stopping={stopping} onResume={resumeAnalysis} onStop={() => void stopAnalysis()} />}
      <MotionRegion show={!!notice}><div className="alert success" role="status">{notice}</div></MotionRegion>
      {!configured && connected && <div className="config-notice"><Info size={17} aria-hidden="true" /><span>{configurationError || '当前服务未启用自动识别。可先上传视频、校准棋盘和手动复盘。'}</span>{!configurationError && <details><summary>如何启用本地识别</summary><p>先停止服务，在项目目录运行 scripts/setup.ps1 -Fusion，再运行 scripts/start-fusion.ps1。训练好的模型已包含，无需 API Key 或重新训练。可选云端接口配置见 README。</p></details>}</div>}
      <div className={`workspace ${reviewIndex || draft ? 'has-editor' : ''}`}>
        <section className="panel source-panel"><div className="panel-heading"><div><Video size={19} /><h2>原始视频</h2></div><span className="panel-tag">视频对照</span></div>
          <VideoPanel task={task} videoRef={video} busy={!!locked} editing={editingCalibration} calibration={calibration} setCalibration={setCalibration} startTime={start}
            onUpload={file => void perform(async () => { setTask(await api.upload(file)); setNotice('视频已上传。请检查棋盘范围，再开始识别。'); })}
            onTime={(time, seeking) => { if (mode === '对照' && task && !draft && !reviewIndex) { setPlaying(false); setBoardJump(seeking); setStep(stepAtVideoTime(task, time)); } }} />
          {task ? <AnimatedDisclosure className="setup-disclosure" title={<><Settings2 size={17} /><span>识别设置</span><small>{task.status === 'analyzing' ? '正在分析' : settingsOpen || editingCalibration ? '已展开' : '已折叠'}</small></>} open={settingsOpen || editingCalibration || task.status === 'analyzing'} onToggle={setSettingsOpen}><div className="setup-section"><div className="section-heading"><h3>识别设置</h3>{!editingCalibration && <button className="text-button" disabled={!!locked} onClick={() => { video.current?.pause(); calibrationBeforeEdit.current = calibration ? structuredClone(calibration) : null; setEditingCalibration(true); }}>调整棋盘范围</button>}</div>
            {editingCalibration && <div className="action-pair"><button className="primary" disabled={!!locked} onClick={() => saveSettings()}>确认棋盘范围</button><button className="secondary" disabled={!!locked} onClick={() => { setCalibration(calibrationBeforeEdit.current); setEditingCalibration(false); }}>取消调整</button></div>}
            {!editingCalibration && task.calibration_confirmed && !settingsChanged && <p className="hint" role="status">棋盘范围已确认，识别和解析使用此范围。</p>}
            <div className="input-pair"><label>开始时间（秒）<input type="number" min="0" max={end} step="0.1" aria-label="开始时间" value={start} disabled={!!locked} onChange={e => setStart(Number(e.target.value))} /></label><label>结束时间（秒）<input type="number" min={start} max={task.metadata.duration_s} step="0.1" aria-label="结束时间" value={end} disabled={!!locked} onChange={e => setEnd(Number(e.target.value))} /></label></div>
            <button className="text-button" disabled={!!locked || editingCalibration} onClick={() => { if (video.current) { video.current.pause(); setStart(Math.round(video.current.currentTime * 1000) / 1000); } }}>使用当前视频画面作为开始时间</button>
            <label>原视频棋盘方向<select aria-label="棋盘方向" value={calibration?.orientation ?? 'normal'} disabled={!!locked} onChange={e => calibration && setCalibration({ ...calibration, orientation: e.target.value as Calibration['orientation'] })}><option value="normal">红方在下</option><option value="rotated_180">红方在上，旋转 180°</option><option value="mirrored">画面左右镜像</option><option value="mirrored_rotated_180">旋转 180° 且镜像</option></select></label>
            <p className="hint">选择任意时间段并检查棋盘范围，即可自动建立起始局面、追踪实际变化。起点遮挡时向前寻找操作前画面；只复刻所选区间。</p>
            <div className="action-pair"><button className="secondary" disabled={!!locked || !!draft} onClick={() => saveSettings()}>保存设置</button><button className="secondary" disabled={!!locked || !configured || !calibration || !!draft} onClick={() => saveSettings('initial')}>{busy ? '正在处理…' : '单独识别起始画面（可选）'}</button></div>
            <button className="primary full" aria-busy={task.status === 'analyzing' || busy} disabled={!!locked || !configured || !calibration} onClick={() => saveSettings('analyze')}>{task.status === 'analyzing' ? '正在分析…' : '自动解析并复刻所选区间'}</button>
            {task.status === 'analyzing' && <button className="stop-analysis full" disabled={stopping} onClick={() => void stopAnalysis()}>{stopping ? '正在停止…' : '停止分析并保留进度'}</button>}
            <div className="cost-status" aria-label="识别用量"><strong>检测与识别记录</strong><div><span>本地检查 {task.analysis_stats?.scanned_frames ?? 0} 帧</span><span>模型请求 {task.analysis_stats?.model_requests ?? 0} 次</span><span>发送 {task.analysis_stats?.images_sent ?? 0} 张图</span><span>已记录 {tokenEntries.length ? recordedTokens.toLocaleString() : '—'} Token</span></div><small>请求与图片数从本次升级后累计；Token 保留历史记录。无变化时不重复请求，短暂疑点会补充图片。金额以供应方账单为准。</small></div>
            {task.run_stats && <div className="cost-status" aria-label="本次分析耗时"><strong>本次分析</strong><div><span>用时 {task.run_stats.elapsed_seconds.toFixed(1)} 秒</span><span>本次模型请求 {task.run_stats.model_requests} 次</span><span>本次发送 {task.run_stats.images_sent} 张图</span>{task.run_stats.backfill_intervals != null && <span>局部回查 {task.run_stats.backfill_intervals} 次 · {(task.run_stats.backfill_seconds ?? 0).toFixed(1)} 秒</span>}</div><small>准备与初始化 {(task.run_stats.initialization_seconds ?? 0).toFixed(1)} 秒 · 主扫描解码 {task.run_stats.decode_seconds.toFixed(1)} 秒 · 主扫描检测 {task.run_stats.opencv_seconds.toFixed(1)} 秒 · 模型等待 {task.run_stats.model_seconds.toFixed(1)} 秒</small></div>}
          {task && <AnimatedDisclosure className="sop" title="复盘操作说明"><p>先检查原视频与四角范围，再识别。点击棋谱可定位原视频；待确认步骤可查看截图并校正。结果自动保存在当前任务中。</p></AnimatedDisclosure>}
          </div></AnimatedDisclosure> : <div className="source-guide"><h3>准备一段完整的对局录屏</h3><p>棋盘完整、镜头固定，方便准确定位。轻度模糊或短时遮挡会结合前后画面分析。</p><div><span>固定视角</span><span>完整棋盘</span><span>连续对局</span></div></div>}
        </section>
        <section className="panel replay-panel"><div className="panel-heading"><div><Grid2X2 size={19} /><h2>棋盘复盘</h2></div><div className="segmented replay-mode" role="group" aria-label="回放模式" data-mode={mode}><span className="mode-indicator" aria-hidden="true" /><button aria-pressed={mode === '演绎'} className={mode === '演绎' ? 'active' : ''} onClick={() => { setMode('演绎'); video.current?.pause(); }}>演绎模式</button><button aria-pressed={mode === '对照'} className={mode === '对照' ? 'active' : ''} onClick={() => { setMode('对照'); setPlaying(false); if (task && video.current && !draft && !reviewIndex) { setBoardJump(true); setStep(stepAtVideoTime(task, video.current.currentTime)); } }}>视频对照</button></div></div>
          <div className={`board-container ${reviewIndex ? 'editing-board' : ''}`}><div className="board-editor"><Board key={task?.id ?? 'empty'} position={display} step={task ? displayStep(task, step) : step} replace={mode === '对照' && boardJump} lastMove={lastMove} inferredMove={task && isPredictedRecord(task.moves[step - 1]) ? lastMove : null} selected={reviewIndex ? picked : null} onSquare={!locked && (draft || reviewIndex) ? paint : undefined} reviews={initialReviews} /><MotionRegion show={!!reviewIndex && !!task} inline className="board-review-presence">{reviewIndex && task && (reviewPosition ? <ReviewPanel key={`${task.id}:${reviewIndex}:${inserting}`} insert={inserting} task={task} index={reviewIndex} position={reviewPosition} picked={picked} busy={!!locked} videoTime={video.current?.currentTime ?? task.start_s} onPreview={setReviewDraft} onClose={() => { setReviewIndex(null); setPicked(null); }} onSavePosition={(position, time, startTime) => void perform(async () => { const saved = await api.correctPosition(task, reviewIndex, position, time, startTime); setTask(saved); setStep(reviewIndex); setReviewIndex(null); setPicked(null); setNotice('局面已保存，后续保留原走法并承接校正结果。'); })} onSave={(moves, time, startTime) => void perform(async () => { if (inserting && moves[0]) {
                const saved = await api.insert(task, reviewIndex, moves[0], time, startTime);
                setTask(saved); setStep(reviewIndex); setReviewIndex(reviewIndex + 1); setPicked(null); setReviewDraft(null);
                setRangeSteps([]); setNotice('遗漏走棋已插入，已有步骤顺延；可以继续补下一手。'); return;
              } const saved = await api.correct(task, reviewIndex, moves, time, startTime); setTask(saved); setStep(isIgnored(saved.moves[reviewIndex - 1]) ? previousStep(saved, reviewIndex) : reviewIndex); setReviewIndex(null); setPicked(null); setNotice(moves.length ? '当前走法已替换，后续承接校正结果；规则问题仅提示。' : '已确认不计入走棋，列表、计数、回放和导出均已跳过。'); })} /> : <div className="alert error">请先确认第 {task.confirmed_prefix_length + 1} 手，再处理后面的步骤。<button className="text-button" onClick={() => setReviewIndex(task.confirmed_prefix_length + 1)}>前往首个疑点</button></div>)}</MotionRegion></div></div>
          <MotionRegion className="inference-notice" show={!!task && isPredictedRecord(task.moves[step - 1])}><p role="status" className="hint">当前步骤采用最大概率推断，后续仍会继续解析；证据充分的步骤自动确认，可随时校正。</p></MotionRegion>
          {!reviewIndex && task?.moves[step - 1]?.violation_note && <p className="rule-warning replay-rule-warning" role="status">{task.moves[step - 1].violation_note}</p>}
          <div className="replay-controls"><button aria-label="回到初始局面" title="回到初始局面" disabled={!display || !!draft} onClick={() => selectStep(0)}><SkipBack size={17} /></button><button aria-label="上一步" disabled={step === 0 || !!draft} onClick={() => task && selectStep(previousStep(task, step))}><ChevronLeft size={17} /><span>上一步</span></button>
            <button className="play-button" aria-label={playing ? '暂停回放' : '自动播放'} disabled={!task || !replayRecords(task).some(r => isAutomaticRecord(r) || isPredictedRecord(r) || r.status === 'corrected') || !!draft || mode === '对照'} onClick={() => { followAnalysis.current = false; setPlaying(!playing); setReviewIndex(null); if (!playing && step >= lastKnown) setStep(0); }}>{playing ? <Pause size={21} fill="currentColor" /> : <Play size={21} fill="currentColor" />}</button>
            <button aria-label="下一步" disabled={!task || step >= lastKnown || !!draft} onClick={() => { if (task) { const next = replayRecords(task).find(r => r.index > step); if (next) selectStep(next.index); } }}><span>下一步</span><ChevronRight size={17} /></button><span className="step-count">第 <b>{task ? displayStep(task, step) : 0}</b> / {total} {task?.rules_enabled ? '步' : '手'}</span>
            <select aria-label="播放步间隔" value={interval} onChange={e => setIntervalSeconds(Number(e.target.value))}>{[1, 2, 3].map(n => <option key={n} value={n}>{n} 秒 / 手</option>)}</select>
          </div>
          {display && <div className="piece-counts" aria-label="棋子数量"><span>{draft ? '待核对局面' : `第${task ? displayStep(task, reviewIndex ?? step) : 0}${task?.rules_enabled ? '步' : '手'}局面`}</span><strong>共 {display.pieces.length} 枚</strong><span>红方 {display.pieces.filter(p => p.color === 'red').length} 枚</span><span>黑方 {display.pieces.filter(p => p.color === 'black').length} 枚</span></div>}
          <div className="board-context"><span className="turn-indicator"><i className={`side-dot ${display?.side_to_move ?? ''}`} />{display ? task?.rules_enabled ? '规则问题仅提示 · 可手动校正' : task?.replay_mode === 'actual' ? '按实际画面还原局面' : `${display.side_to_move === 'red' ? '红方' : '黑方'}走棋` : '等待棋局'}</span></div>
          <div className="board-actions"><span>{confirmedMoves} 手已确认{total > confirmedRecords && ` · ${total - confirmedRecords} 手需人工审核`}{!!intervals.length && ` · ${intervals.length} 段视频待核对`}</span><div>{task && <button className="text-button" disabled={!!locked} onClick={editInitial}>设置初始棋局</button>}{step > 0 && task && <button className="text-button" disabled={!!locked || !!draft} onClick={() => { setPlaying(false); setPicked(null); setInserting(false); setReviewIndex(step); }}>校正此步</button>}{step > 0 && task && <button className="text-button" disabled={!!locked || !!draft} onClick={() => { setPlaying(false); setPicked(null); setInserting(true); setReviewIndex(reviewIndex ?? step); }}>在此步前插入</button>}{task?.initial_position && <button className="text-button" disabled={!!locked || !!draft} onClick={() => { setPlaying(false); setStep(lastKnown); setPicked(null); setInserting(true); setReviewIndex(task.moves.length + 1); }}>末尾追加走棋</button>}</div></div>
          <MotionRegion show={!!draft && !!task}>{draft && task && <section className="review-panel"><div className="section-heading"><h3>设置初始棋局</h3><button className="text-button" onClick={() => setDraft(null)}>取消</button></div><p className="hint">选择棋子后点击交叉点放置；选择“移除”删除该位置棋子。允许特殊摆局数量，保存时检查重复位置和将帅位置。</p>
            {initialReviews.length > 0 && <div className="initial-review-warning" role="status"><strong>已展示可识别部分，以下 {initialReviews.length} 处需要核对</strong><p>橙色虚线为待核对位置；问号表示棋种未知，方框表示有无棋子尚不确定。候选棋种和模型评分不代表已确认正确。</p><ul>{initialReviews.map(r => <li key={`${r.square.file}-${r.square.rank}`}>第{r.square.file + 1}列第{r.square.rank + 1}行 · {r.state === 'occupancy' ? '占位未知' : `${r.color === 'red' ? '红' : r.color === 'black' ? '黑' : ''}${r.prediction && r.color ? pieceText[r.color][r.prediction] : '棋种未知'}`} · {r.reason}</li>)}</ul></div>}
            {task.initial_frame_s !== null && task.initial_evidence_ids.length > 0 && <details className="initial-evidence"><summary>查看识别依据 · 视频 {timeText(task.initial_frame_s)}{task.initial_frame_s < task.start_s && '（前置画面，仅用于初始化）'}</summary><a href={`/api/tasks/${task.id}/evidence/${task.initial_evidence_ids[0]}`} target="_blank" rel="noreferrer"><img src={`/api/tasks/${task.id}/evidence/${task.initial_evidence_ids[0]}`} alt="初始局面识别使用的棋盘截图" /></a></details>}
            {task.initial_candidates.length > 1 && <label>模型候选局面<select aria-label="模型候选局面" value={candidateIndex} disabled={!!locked} onChange={e => { const index = Number(e.target.value); setCandidateIndex(index); setDraft(structuredClone(task.initial_candidates[index])); }}>{task.initial_candidates.map((p, index) => <option key={index} value={index}>候选 {index + 1} · {p.pieces.length} 枚 · {p.side_to_move === 'red' ? '红方' : '黑方'}下一手</option>)}</select></label>}
            <p className="hint">{task.rules_enabled ? '复盘按象棋规则校验走法和轮次，请核对起始局面及下一方。' : '实际局面复刻不依赖下一方或合法走法。这里可以校正棋子位置、类型和数量；初始识别存在真实歧义时才需要核对。'}</p>
            <div className="input-pair"><label>放置棋子<select aria-label="放置棋子" value={brush} onChange={e => setBrush(e.target.value)}>{(['red', 'black'] as Color[]).flatMap(color => (Object.keys(pieceText[color]) as PieceKind[]).map(kind => <option key={`${color}:${kind}`} value={`${color}:${kind}`}>{color === 'red' ? '红' : '黑'}{pieceText[color][kind]}</option>))}<option value="remove">移除棋子</option></select></label><label>下一方<select aria-label="下一方" value={draft.side_to_move} onChange={e => setDraft({ ...draft, side_to_move: e.target.value as Color })}><option value="red">红方</option><option value="black">黑方</option></select></label></div>
            {settingsChanged && <p className="hint">范围或四角已修改，保存时会与当前手动棋局一并提交，无需重新识别。</p>}
            <div className="action-pair"><button className="secondary" onClick={() => setDraft(opening())}>恢复标准开局</button><button className="primary" disabled={!!locked} onClick={() => saveSettings('confirm')}>确认并保存初始棋局</button></div>
          </section>}</MotionRegion>

        </section>
        <section className="panel steps-panel">
          <MoveList task={task} step={step} onSelect={selectStep} selecting={selectingRange} boundaries={rangeSteps}
            onRecheck={chooseRange} disabled={!!locked || !!draft || !!reviewIndex || settingsChanged}
            onCancelSelection={() => { setSelectingRange(false); setRangeSteps([]); }} />
        </section>
      </div>
      <MotionRegion show={!!task}>{task && <div className="studio-progress-grid"><section className="analysis-status studio-progress" aria-label="分析进度"><div role="status"><span className={task.status === 'analyzing' ? 'pulse-dot' : 'small-dot'} /><strong key={task.stage} className="stage-label">{task.stage}</strong></div>
              <MotionRegion show={!task.initial_position}><p role="status">起始局面尚未建立，无法生成走棋步骤。{locked ? '请先停止分析，再核对初始棋局；已有扫描结果保留。' : '请核对标记的位置并保存初始棋局，再开始解析。'}</p><button className="secondary" disabled={!!locked} onClick={editInitial}>核对初始棋局</button></MotionRegion>
              <div><strong>视频扫描</strong><span>{Math.round(scanProgress)}%</span></div><div className="progress-track" role="progressbar" aria-label="视频扫描进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(scanProgress)}><i style={{ transform: `scaleX(${scanProgress / 100})` }} /></div><small>已扫描至 {timeText(scanTime)} / {timeText(task.end_s)} · 保存 {task.scan_events ?? 0} 个疑似变化事件</small>
              <div><strong>局面处理</strong><span>{Math.round(progress)}%</span></div><div className="progress-track" role="progressbar" aria-label="局面处理进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)}><i style={{ transform: `scaleX(${progress / 100})` }} /></div><small>已处理至 {timeText(task.recognition_video_s || task.processed_video_s)} · 最近确认落点 {timeText(confirmedTime)} · {intervals.length} 段待核对区间 · 待处理 {Math.max(task.pending_event_count ?? 0, task.recovery?.pending_request ? 1 : 0)} 个</small>
              <MotionRegion show={task.tracking_state === 'recovering'}><p role="status">从 {timeText(task.tracking_since_s ?? confirmedTime)} 起{task.status === 'analyzing' || task.status === 'waiting_service' ? '正在重新同步' : '的画面仍需核对'}：{task.tracking_message ?? '等待连续清晰画面'}。已有棋谱保留，未确定区间继续保留供核对。</p></MotionRegion>
              {task.rebuild_video_s != null && <p>正在重建至 {timeText(task.rebuild_video_s)}，目标 {timeText(task.rebuild_target_s ?? task.processed_video_s)}</p>}
              {task.recovery?.request_started_at && <p>正在{task.recovery.phase === 'tls' ? '建立安全连接' : task.recovery.phase === 'upload' ? '发送截图' : task.recovery.phase === 'response' ? '等待模型响应' : '连接模型服务'}，已等待 {requestSeconds} 秒；已有扫描结果保留。</p>}
              {task.status === 'analyzing' && task.recovery?.heartbeat_at && <small>后台最后活动：{Math.max(0, Math.floor(clock / 1000 - task.recovery.heartbeat_at))} 秒前</small>}
            </section>
          <RecheckPanel key={task.id} task={task} busy={busy || !!reviewIndex || !!draft || settingsChanged} onSelect={selectStep}
            onCancel={() => void perform(async () => { setTask(await api.cancelRecheck(task)); })}
            onApply={() => {
              void perform(async () => { const latest = await api.get(task.id); const saved = await api.applyRecheck(latest); setTask(saved); setStep(saved.recheck?.left_index ?? 0); setReviewIndex(null); setReviewDraft(null); setDraft(null); setPicked(null); setInserting(false); setRangeSteps([]); setNotice('区间棋谱已更新，后续回放与导出使用修复后的顺序。'); }); }} />
      </div>}</MotionRegion>
      {task && <AnimatedDisclosure className="studio-export-details" title={<><Clapperboard size={16} /><span>对局录像与导出记录</span></>}><RecordingPanel key={task.id} task={task} intervalS={interval} toolbarTarget={exportTarget} /></AnimatedDisclosure>}
      <footer><span>棋局复刻 <i>·</i> 从影像中还原棋局</span><span>视频与复盘结果保存在本机</span></footer>
    </main>
  </div>;
}
