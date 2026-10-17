import json
import math
import os
import queue
import re
import shutil
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import av
import cv2
import numpy as np

from .schemas import Task
from .storage import RevisionConflict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_recording_playlist(task: Task, interval_s: int) -> dict:
    if interval_s not in (1, 2, 3):
        raise ValueError('请选择 1、2 或 3 秒每手')
    if task.status in ('analyzing', 'waiting_service', 'action_required') or task.pending_turn is not None:
        raise ValueError('请等待分析结束并确认全部步骤后再导出')
    if task.initial_position is None:
        raise ValueError('请先确认并保存初始棋局')
    if task.processed_video_s + .001 < task.end_s:
        raise ValueError('所选区间尚未完整分析，请完成分析后再导出')
    unresolved = [r.index for r in task.moves if r.status not in ('auto_verified', 'corrected')]
    if unresolved:
        raise ValueError('请先校正第 ' + '、'.join(map(str, unresolved)) + ' 步，再导出完整录像')
    records = []
    for record in task.moves:
        if not task.start_s <= record.video_end_s <= task.end_s:
            raise ValueError(f'第 {record.index} 步时间不在已保存选区内，请先校正')
        if record.change_type == 'review' and record.move is None:
            continue
        if record.position_after is None:
            raise ValueError(f'第 {record.index} 步缺少确认局面，请先校正')
        records.append(record.model_dump(by_alias=True))
    duration = (len(records) + 1) * interval_s
    return {'initial_position': task.initial_position.model_dump(by_alias=True), 'records': records,
            'interval_s': interval_s, 'duration_s': duration, 'total_frames': duration * 60}


def _stop_renderer(process):
    if process.poll() is not None:
        return
    # 只终止本次导出创建的进程树，包括它启动的无界面 Chrome。
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    else:
        process.terminate()
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)


def encode_recording(manifest_path: Path, output_path: Path, progress, render_url: str, on_process=None):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    node = shutil.which('node')
    if node is None:
        raise RuntimeError('未找到 Node.js，请检查本项目运行环境')
    frame_count = 0
    error_log = manifest_path.parent / 'renderer-error.log'
    with error_log.open('wb') as errors:
        process = subprocess.Popen([node, str(PROJECT_ROOT / 'scripts/render-recording.cjs'), str(manifest_path), render_url],
                                   cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors)
        packets = queue.Queue(maxsize=2)
        frames = queue.Queue(maxsize=2)
        stopping = threading.Event()

        def publish(target, item):
            while not stopping.is_set():
                try:
                    target.put(item, timeout=.1)
                    return True
                except queue.Full:
                    pass
            return False

        def read_packets():
            try:
                while not stopping.is_set() and (line := process.stdout.readline()):
                    packet = json.loads(line)
                    image = process.stdout.read(packet['bytes'])
                    if len(image) != packet['bytes']:
                        raise RuntimeError('棋盘画面流中断，请重新导出')
                    if not publish(packets, (image, packet['repeat'])):
                        return
            except Exception as exc:
                publish(packets, exc)
            finally:
                publish(packets, None)

        def decode_frames():
            try:
                while not stopping.is_set():
                    try:
                        item = packets.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    image, repeat = item
                    pixels = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
                    if pixels is None or pixels.shape[:2] != (1192, 1080):
                        raise RuntimeError('棋盘画面尺寸不正确，无法生成录像')
                    if not publish(frames, (pixels, repeat)):
                        return
            except Exception as exc:
                publish(frames, exc)
            finally:
                publish(frames, None)

        workers = (
            threading.Thread(target=read_packets, name='recording-frame-reader', daemon=True),
            threading.Thread(target=decode_frames, name='recording-frame-decoder', daemon=True),
        )
        try:
            if on_process is not None:
                on_process(process)
            for worker in workers:
                worker.start()
            with av.open(str(output_path), 'w', format='mp4', options={'movflags': '+faststart'}) as container:
                stream = container.add_stream('libx264', rate=60)
                stream.width, stream.height, stream.pix_fmt = 1080, 1192, 'yuv420p'
                stream.options = {'crf': '18', 'preset': 'fast'}
                while True:
                    item = frames.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    pixels, repeat = item
                    frame = av.VideoFrame.from_numpy_buffer(pixels, format='bgr24')
                    frame.time_base = Fraction(1, 60)
                    for _ in range(repeat):
                        frame.pts = frame_count
                        for encoded in stream.encode(frame):
                            container.mux(encoded)
                        frame_count += 1
                    progress(frame_count)
                code = process.wait(timeout=10)
                if code != 0:
                    raise RuntimeError('棋盘渲染失败：' + error_log.read_text(encoding='utf-8', errors='replace')[-700:])
                if frame_count != manifest['total_frames']:
                    raise RuntimeError('录像帧数不完整，请重新导出')
                for encoded in stream.encode():
                    container.mux(encoded)
        finally:
            stopping.set()
            _stop_renderer(process)
            for worker in workers:
                if worker.ident is not None:
                    worker.join(timeout=5)
            process.stdout.close()


class RecordingManager:
    def __init__(self, root: Path, output_root: Path, render_url='http://127.0.0.1:5278/recording-render'):
        self.root, self.output_root = Path(root).resolve(), Path(output_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.render_url = render_url
        self.lock = threading.RLock()
        self.jobs, self.threads, self.processes = {}, {}, {}
        self.closing = threading.Event()
        for path in self.root.glob('*/status.json'):
            job = json.loads(path.read_text(encoding='utf-8'))
            self.jobs[job['id']] = job
            if job['status'] in ('queued', 'running'):
                job.update(status='failed', stage='导出中断', error='服务重启导致导出中断，请重新导出', file_path=None)
                self._persist(job)

    def _persist(self, job):
        path = self.root / job['id'] / 'status.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return dict(self.jobs[job_id])

    def latest(self, task_id):
        with self.lock:
            jobs = [job for job in self.jobs.values() if job['task_id'] == task_id]
            return dict(max(jobs, key=lambda job: job['created_at'])) if jobs else None

    def start(self, task: Task, revision: int, interval_s: int):
        if revision != task.revision:
            raise RevisionConflict('棋局已更新，请刷新核对后再导出')
        payload = build_recording_playlist(task, interval_s)
        with self.lock:
            if self.closing.is_set():
                raise ValueError('服务正在关闭，请重新启动后再导出')
            for job in self.jobs.values():
                if job['status'] in ('queued', 'running'):
                    if (job['task_id'], job['revision'], job['interval_s']) == (task.id, revision, interval_s):
                        return dict(job)
                    raise ValueError('已有录像正在生成，请完成后再导出')
            job_id = uuid4().hex
            directory = self.root / job_id
            directory.mkdir()
            (directory / 'manifest.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            now = datetime.now(timezone(timedelta(hours=8)))
            stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', Path(task.filename).stem).strip(' .')[:80] or '象棋对局'
            filename = f'{stem}_{task.start_s:.3f}-{task.end_s:.3f}s_{now:%Y%m%d_%H%M%S}_{job_id[:8]}.mp4'
            job = {'id': job_id, 'task_id': task.id, 'revision': revision, 'interval_s': interval_s, 'status': 'queued',
                   'stage': '等待生成', 'progress': 0, 'completed_frames': 0, 'total_frames': payload['total_frames'],
                   'duration_s': payload['duration_s'], 'width': 1080, 'height': 1192, 'fps': 60,
                   'filename': filename, 'file_path': None, 'error': None, 'created_at': now.isoformat()}
            self.jobs[job_id] = job
            self._persist(job)
            thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
            self.threads[job_id] = thread
            thread.start()
            return dict(job)

    def _run(self, job_id):
        partial = self.root / job_id / 'video.partial.mp4'
        try:
            with self.lock:
                job = self.jobs[job_id]
                if self.closing.is_set():
                    raise RuntimeError('服务停止导致导出中断，请重新导出')
                job.update(status='running', stage='正在生成棋盘录像')
                self._persist(job)
            def update(frames):
                if self.closing.is_set():
                    raise RuntimeError('服务停止导致导出中断，请重新导出')
                with self.lock:
                    job.update(completed_frames=frames, progress=min(99, math.floor(frames / job['total_frames'] * 99)))
                    self._persist(job)
            def register(process):
                with self.lock:
                    self.processes[job_id] = process
                if self.closing.is_set():
                    _stop_renderer(process)
            encode_recording(self.root / job_id / 'manifest.json', partial, update, self.render_url, register)
            with self.lock:
                if self.closing.is_set():
                    raise RuntimeError('服务停止导致导出中断，请重新导出')
                final = self.output_root / job['filename']
                if final.exists():
                    raise RuntimeError('同名录像已存在，请重新导出，不会覆盖旧文件')
                partial.rename(final)
                job.update(status='completed', stage='录像已保存', progress=100, file_path=str(final))
                self._persist(job)
        except Exception as exc:
            partial.unlink(missing_ok=True)
            with self.lock:
                job = self.jobs[job_id]
                error = '服务停止导致导出中断，请重新导出' if self.closing.is_set() else str(exc)
                job.update(status='failed', stage='导出失败', error=error, file_path=None)
                self._persist(job)
        finally:
            with self.lock:
                self.processes.pop(job_id, None)

    def close(self):
        self.closing.set()
        with self.lock:
            processes, threads = list(self.processes.values()), list(self.threads.values())
        for process in processes:
            _stop_renderer(process)
        for thread in threads:
            thread.join(timeout=10)
