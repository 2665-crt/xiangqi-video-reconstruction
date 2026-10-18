import os
import json
import re
import threading
import multiprocessing
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from .calibration import detect_calibration, rectify_board, validate_calibration, refine_calibration
from .schemas import CorrectionRequest, InitialPieceReview, InitialRecognitionRequest, InsertRequest, Position, RecordingRequest, RecheckActionRequest, RecheckRequest, RecheckState, SetupRequest, Task
from .recording import RecordingManager
from .analysis import analyze_task, correct_interval, revalidate_suffix
from .vision import VisionClient
from .storage import RevisionConflict, TaskStore
from .recovery import RecoveryCoordinator, prepare_recovery, run_saved_operation
from .video import iter_frames, probe_video, save_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def run_analysis_worker(task_id: str, data_root: str, local_only: bool = False, generation=None, lease_token=None):
    store = TaskStore(Path(data_root))
    task = store.get(task_id)
    if generation is not None and (task.recovery is None or task.recovery.generation != generation or task.recovery.lease_token != lease_token or not task.recovery.auto_run):
        return
    if store.get(task_id).recovery is not None:
        run_saved_operation(task_id, store)
    else:
        analyze_task(task_id, store, None if local_only else VisionClient())


def create_app(data_root: Path | None = None, recording_root: Path | None = None):
    store = TaskStore(data_root or PROJECT_ROOT / "data")
    recordings = RecordingManager(store.root / 'recordings', recording_root or PROJECT_ROOT / '录屏存放')
    workers = {}
    recheck_threads = {}
    analysis_lock = threading.Lock()
    vision_runtime_lock = threading.Lock()

    def stop_worker(task_id, manual=True):
        task = store.get(task_id)
        if task.recovery is not None and task.recovery.auto_run:
            def cancel(saved):
                saved.recovery.auto_run = not manual
                saved.recovery.lease_token = None
                saved.status = 'interrupted' if manual else 'waiting_service'
                saved.stage = '分析已停止，已保留进度和棋谱' if manual else '后台关闭，任务已保存，将在启动后自动恢复'
                saved.error = None
            task = store.update_recovery(task.id, task.recovery.generation, task.recovery.lease_token, cancel)
        process = workers.pop(task_id, None)
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        task = store.get(task_id)
        if task.status == "analyzing":
            if task.pending_turn is not None:
                pending = task.pending_turn
                pending.index, pending.status, pending.position_after = len(task.moves) + 1, 'needs_review', None
                pending.reason = '分析已停止，本回合尚未完成确认，请人工审核'
                task.moves.append(pending)
                task.pending_turn = None
            task.status, task.stage, task.error = "interrupted", "分析已停止，已保留进度和棋谱", None
            task = store.save(task, task.revision)
        return task

    def launch_recovery(task):
        process = multiprocessing.get_context('spawn').Process(target=run_analysis_worker,
            args=(task.id, str(store.root), task.recovery.local_only),
            kwargs={'generation': task.recovery.generation, 'lease_token': task.recovery.lease_token}, daemon=False)
        process.start()
        return process

    recovery = RecoveryCoordinator(store, workers, analysis_lock, launch_recovery)

    @asynccontextmanager
    async def lifespan(app):
        store.recover_interrupted()
        recovery.start()
        yield
        for task_id in list(recheck_threads):
            current = store.get(task_id)
            if current.recheck is not None and current.recheck.status == 'running':
                current.recheck.status, current.recheck.message = 'cancelled', '后台已关闭；候选与原棋谱保留，可重新选择区间检测'
                store.save(current, current.revision)
        recovery.close()
        for task_id in list(workers):
            stop_worker(task_id, manual=False)
        recordings.close()

    app = FastAPI(title="象棋视频复盘", lifespan=lifespan)
    app.state.store = store
    app.state.analysis_workers = workers
    app.state.recordings = recordings
    app.state.recovery = recovery

    def get_task(task_id):
        try:
            return store.get(task_id)
        except KeyError:
            raise HTTPException(404, "任务不存在")

    def save_task(task, revision):
        try:
            return store.save(task, revision)
        except RevisionConflict:
            raise HTTPException(409, "任务已更新，请刷新后重试")

    def require_editable(task):
        if task.recheck is not None and task.recheck.status == 'running':
            raise HTTPException(409, '请先停止区域重检，再修改棋谱或分析设置')

    def require_vision_runtime():
        if os.getenv('VISION_FUSION') == '1':
            try:
                from .fusion import local_fusion_classifier
                with vision_runtime_lock:
                    local_fusion_classifier()
            except ModuleNotFoundError as exc:
                raise HTTPException(503, f'本地融合识别环境缺少 {exc.name}，请使用 scripts/start-fusion.ps1 启动完整识别环境。') from None

    @app.get("/api/health")
    def health():
        try:
            require_vision_runtime()
        except HTTPException as exc:
            return {'status': 'error', 'vision_configured': False, 'detail': exc.detail}
        return {"status": "ok", "vision_configured": os.getenv('VISION_FUSION') == '1' or all(os.getenv(name) for name in ("VISION_API_BASE", "VISION_MODEL", "VISION_API_KEY"))}

    @app.get("/api/tasks")
    def list_tasks():
        from .records import movement_records
        return [{"id": t.id, "filename": t.filename, "status": t.status,
                 "steps": len(movement_records(t))} for t in store.list_tasks()]

    @app.post("/api/tasks", status_code=201, response_model=Task)
    async def upload(file: UploadFile):
        if not (file.filename or "").lower().endswith(".mp4"):
            raise HTTPException(422, "第一版请上传 MP4 视频")
        uploads = store.root / "uploads"
        uploads.mkdir(exist_ok=True)
        temporary = uploads / f"{uuid4().hex}.mp4"
        size = 0
        try:
            with temporary.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 2 * 1024 ** 3:
                        raise HTTPException(413, "视频超过 2 GiB 上限")
                    output.write(chunk)
            try:
                metadata = probe_video(temporary)
            except ValueError as exc:
                raise HTTPException(422, str(exc))
            task = store.create(Path(file.filename).name, metadata)
            task.replay_mode = 'actual'
            task.rules_enabled = True
            task.initial_turn_known = False
            directory = store.task_dir(task.id)
            temporary.replace(directory / "source.mp4")
            _, preview = next(iter_frames(directory / "source.mp4", 0, metadata.duration_s))
            evidence = directory / "evidence"
            evidence.mkdir()
            save_image(evidence / "preview.jpg", preview)
            task.calibration = detect_calibration([preview])
            task.initial_evidence_ids = ["preview.jpg"]
            # Upload is the creation operation; the first client-visible revision stays zero.
            with __import__("sqlite3").connect(store.db) as con:
                con.execute("UPDATE tasks SET payload=? WHERE id=?", (task.model_dump_json(by_alias=True), task.id))
            return task
        finally:
            await file.close()
            temporary.unlink(missing_ok=True)

    @app.get("/api/tasks/{task_id}", response_model=Task)
    def task_detail(task_id: str):
        with analysis_lock:
            task = get_task(task_id)
            process = workers.get(task_id)
            if task.status == "analyzing" and process is not None and not process.is_alive():
                process.join(timeout=0)
                del workers[task_id]
                if task.recovery is not None and task.recovery.auto_run:
                    from .recovery import defer_operation
                    from .vision import VisionRequestError
                    task = defer_operation(store, task, VisionRequestError('分析进程意外退出，任务已保存，将自动恢复', 'WorkerExited', 'worker'))
                else:
                    task.status, task.stage, task.error = "failed", "分析进程已退出", "分析进程意外退出，可以重新开始"
                    task = save_task(task, task.revision)
            if task.rules_enabled and task.moves and task.status not in ('analyzing', 'waiting_service'):
                from .turns import revalidate_turns
                original_status, original_stage = task.status, task.stage
                task = revalidate_turns(task)
                if original_status not in ('ready', 'needs_review'):
                    task.status, task.stage = original_status, original_stage
            return task

    @app.get("/api/tasks/{task_id}/video")
    def video(task_id: str):
        get_task(task_id)
        return FileResponse(store.task_dir(task_id) / "source.mp4", media_type="video/mp4")

    @app.post('/api/tasks/{task_id}/recordings', status_code=202)
    def export_recording(task_id: str, request: RecordingRequest):
        task = get_task(task_id)
        try:
            return recordings.start(task, request.revision, request.interval_s)
        except RevisionConflict as exc:
            raise HTTPException(409, str(exc))
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get('/api/tasks/{task_id}/recordings/latest')
    def latest_recording(task_id: str):
        get_task(task_id)
        return recordings.latest(task_id)

    @app.get('/api/recordings/{job_id}')
    def recording_status(job_id: str):
        try:
            return recordings.get(job_id)
        except KeyError:
            raise HTTPException(404, '录像任务不存在')

    @app.get('/api/recordings/{job_id}/video')
    def recording_video(job_id: str):
        job = recording_status(job_id)
        if job['status'] != 'completed' or not job['file_path'] or not Path(job['file_path']).is_file():
            raise HTTPException(409, '录像尚未生成成功，请等待完成或重新导出')
        return FileResponse(job['file_path'], media_type='video/mp4', filename=job['filename'], content_disposition_type='inline')

    @app.get("/api/tasks/{task_id}/frame")
    def frame_preview(task_id: str, time_s: float = Query(ge=0)):
        task = get_task(task_id)
        if time_s >= task.metadata.duration_s:
            raise HTTPException(422, "预览时间必须在视频时长内")
        frames = iter_frames(store.task_dir(task.id) / "source.mp4", time_s, task.metadata.duration_s)
        try:
            _, frame = next(frames)
        except StopIteration:
            raise HTTPException(422, "该时间没有可用画面，请稍向前调整")
        finally:
            frames.close()
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise HTTPException(422, "视频画面无法编码")
        return Response(encoded.tobytes(), media_type="image/jpeg")

    @app.get("/api/tasks/{task_id}/evidence/{evidence_id}")
    def evidence(task_id: str, evidence_id: str):
        get_task(task_id)
        if not re.fullmatch(r"[a-z0-9_-]+\.jpg", evidence_id):
            raise HTTPException(404, "证据不存在")
        path = store.task_dir(task_id) / "evidence" / evidence_id
        if not path.is_file():
            raise HTTPException(404, "证据不存在")
        return FileResponse(path, media_type="image/jpeg")

    @app.patch("/api/tasks/{task_id}/setup", response_model=Task)
    def setup(task_id: str, request: SetupRequest):
        task = get_task(task_id)
        require_editable(task)
        if task.status == "analyzing" or (task.recovery is not None and task.recovery.auto_run):
            raise HTTPException(409, "请先停止分析或自动恢复，再修改设置")
        if not 0 <= request.start_s < request.end_s <= task.metadata.duration_s:
            raise HTTPException(422, "分析范围必须在视频时长内，且结束时间晚于开始时间")
        if request.calibration:
            try:
                validate_calibration(request.calibration, task.metadata.width, task.metadata.height)
            except ValueError as exc:
                raise HTTPException(422, str(exc))
        changed_range = (task.start_s, task.end_s, task.calibration) != (request.start_s, request.end_s, request.calibration)
        task.start_s, task.end_s, task.calibration = request.start_s, request.end_s, request.calibration
        task.calibration_confirmed = request.calibration is not None
        if changed_range:
            task.moves, task.confirmed_prefix_length = [], 0
            task.pending_turn = None
            task.evidence_end_s = None
            task.processed_video_s = task.start_s
            task.initial_position = request.initial_position
            task.context_position = None
            task.initial_candidates, task.initial_frame_s = [], None
            task.initial_draft, task.initial_piece_reviews = None, []
            task.initial_evidence_ids = ["preview.jpg"]
            task.scan_progress_s = task.recognition_video_s = task.start_s
            task.scan_complete = False
            task.scan_events = task.recognition_events = task.pending_event_count = 0
            task.resume_video_s = task.rebuild_video_s = task.rebuild_target_s = None
        elif request.initial_position is not None and request.initial_position != task.initial_position:
            task.initial_position = request.initial_position
            task.context_position, task.initial_frame_s = None, None
            if task.moves:
                task = revalidate_suffix(task, 1)
        if request.initial_position is not None:
            task.initial_draft, task.initial_piece_reviews = None, []
            task.initial_turn_known = True
        task.status, task.stage, task.error = "uploaded", "设置已保存", None
        task.recovery = None
        if task.moves:
            task = revalidate_suffix(task, 1)
        return save_task(task, request.revision)

    @app.post("/api/tasks/{task_id}/recognize-initial", response_model=Task)
    def recognize_initial(task_id: str, request: InitialRecognitionRequest):
        task = get_task(task_id)
        require_editable(task)
        if task.revision != request.revision:
            raise HTTPException(409, "任务已更新，请刷新后重试")
        if not task.calibration:
            raise HTTPException(422, "请先定位棋盘四角并保存设置")
        require_vision_runtime()
        vision = VisionClient()
        if not vision.configured:
            vision.close()
            raise HTTPException(409, "请先在后端配置视觉模型")
        if not analysis_lock.acquire(timeout=1):
            vision.close()
            raise HTTPException(409, "已有识别正在进行，请稍后重试")
        task = get_task(task_id)
        if task.revision != request.revision:
            analysis_lock.release()
            vision.close()
            raise HTTPException(409, '任务已更新，请刷新后重试')
        if any(process.is_alive() for process in workers.values()):
            analysis_lock.release()
            vision.close()
            raise HTTPException(409, "已有视频正在分析，请先停止分析")
        try:
            task = prepare_recovery(task, store, operation='initial')
            from .initial import run_initial_operation
            return run_initial_operation(task, store, vision)
        finally:
            analysis_lock.release()

    @app.post('/api/tasks/{task_id}/resume', response_model=Task)
    def resume(task_id: str, request: InitialRecognitionRequest):
        require_editable(get_task(task_id))
        get_task(task_id)
        try:
            return recovery.request_resume(task_id, request.revision)
        except (ValueError, RevisionConflict) as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/tasks/{task_id}/analyze", status_code=202, response_model=Task)
    def analyze(task_id: str, local_only: bool = Query(False), actual: bool = Query(True)):
        task = get_task(task_id)
        require_editable(task)
        if task.recovery is not None and task.recovery.auto_run:
            raise HTTPException(409, '任务正在分析或自动恢复，请先停止后再重新分析')
        if not local_only:
            require_vision_runtime()
        vision = None if local_only else VisionClient()
        if vision is not None and not vision.configured:
            vision.close()
            raise HTTPException(409, "请先在后端配置视觉模型，上传与人工复盘仍可使用")
        if task.initial_position is None and not actual:
            if vision is not None:
                vision.close()
            raise HTTPException(409, "请先核对并保存初始棋局，再分析视频变化")
        if not analysis_lock.acquire(timeout=1):
            if vision is not None:
                vision.close()
            raise HTTPException(409, "已有视频正在分析，请完成后再开始")
        try:
            if any(process.is_alive() for process in workers.values()):
                raise HTTPException(409, "已有视频正在分析，请先停止分析")
            for old_id, process in list(workers.items()):
                process.join(timeout=0)
                del workers[old_id]
            task.status, task.stage, task.error = "analyzing", "本地复刻（禁止云端请求）" if local_only else "开始分析", None
            task.replay_mode = 'actual' if actual else 'legal'
            task.rules_enabled = actual
            task.moves, task.confirmed_prefix_length = [], 0
            task.pending_turn = None
            task.evidence_end_s = None
            task.processed_video_s = task.start_s
            task.recognition_video_s = task.start_s
            task.recognition_events = 0
            task.pending_event_count = task.scan_events
            task.resume_video_s = task.rebuild_video_s = task.rebuild_target_s = None
            saved = prepare_recovery(task, store, local_only=local_only)
            process = multiprocessing.get_context("spawn").Process(target=run_analysis_worker,
                args=(task_id, str(store.root), local_only),
                kwargs={'generation': saved.recovery.generation, 'lease_token': saved.recovery.lease_token}, daemon=False)
            try:
                process.start()
            except Exception:
                saved.status, saved.stage, saved.error = "failed", "分析未启动", "无法启动分析进程，请检查本地运行环境"
                save_task(saved, saved.revision)
                raise HTTPException(500, "无法启动分析进程，请检查本地运行环境") from None
            workers[task_id] = process
            return saved
        finally:
            analysis_lock.release()
            if vision is not None:
                vision.close()

    @app.post("/api/tasks/{task_id}/stop", response_model=Task)
    def stop(task_id: str):
        task = get_task(task_id)
        with analysis_lock:
            if task.status == "analyzing" and task_id not in workers:
                raise HTTPException(409, "起始画面识别正在进行，请等待这一次请求结束")
            return stop_worker(task_id)

    @app.patch("/api/tasks/{task_id}/moves/{index}", response_model=Task)
    def correction(task_id: str, index: int, request: CorrectionRequest):
        task = get_task(task_id)
        require_editable(task)
        if task.recovery is not None and task.recovery.auto_run:
            raise HTTPException(409, '请先停止分析或自动恢复，再校正棋局')
        if task.status == "analyzing" and not task.rules_enabled:
            raise HTTPException(409, "分析期间不能校正")
        if task.status == 'analyzing' and index > len(task.moves):
            raise HTTPException(422, '请等待该步骤识别完成后再确认')
        try:
            if request.position is not None:
                from .turns import correct_turn
                fixed = correct_turn(task, index, [], request.video_time_s, request.position)
            else:
                actions = request.moves if request.moves is not None else [request.move]
                if len(actions) <= 1:
                    from .turns import correct_turn
                    fixed = correct_turn(task, index, actions, request.video_time_s)
                else:
                    fixed = correct_interval(task, index, actions, request.video_time_s)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        if request.action_start_s is not None:
            record = fixed.moves[index - 1]
            preceding = (task.moves[index - 2].action_end_s if task.moves[index - 2].action_end_s is not None else task.moves[index - 2].video_end_s) if index > 1 else task.start_s
            if not preceding <= request.action_start_s <= record.video_end_s:
                raise HTTPException(422, '动作开始时间必须位于前一步落定与本步落定之间')
            record.action_start_s, record.action_end_s, record.timing_status = request.action_start_s, record.video_end_s, 'observed'
        fixed.recovery = None
        fixed.recheck = None
        return save_task(fixed, task.revision if task.status == 'analyzing' else request.revision)

    @app.post('/api/tasks/{task_id}/moves/{index}/insert', response_model=Task)
    def insert_move(task_id: str, index: int, request: InsertRequest):
        task = get_task(task_id)
        require_editable(task)
        if task.status == 'analyzing' or task.recovery is not None and task.recovery.auto_run:
            raise HTTPException(409, '请先停止分析，再插入遗漏走棋')
        if task.revision != request.revision:
            raise HTTPException(409, '任务已更新，请刷新后重试')
        try:
            from .turns import insert_turn
            fixed = insert_turn(task, index, request.move, request.video_time_s, request.action_start_s)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        fixed.recovery = None
        return save_task(fixed, request.revision)

    @app.post('/api/tasks/{task_id}/recheck', status_code=202, response_model=Task)
    def start_recheck(task_id: str, request: RecheckRequest):
        from .recheck import range_bounds, run_recheck
        with analysis_lock:
            task = get_task(task_id)
            require_editable(task)
            if task.status == 'analyzing' or task.recovery is not None and task.recovery.auto_run:
                raise HTTPException(409, '请先停止当前分析，再选择区域重检')
            if task.revision != request.revision:
                raise HTTPException(409, '任务已更新，请刷新后重试')
            if task.calibration is None or not (store.task_dir(task.id) / 'source.mp4').is_file():
                raise HTTPException(422, '区域重检需要原视频与已保存的棋盘校准')
            try:
                left, right, start, end = range_bounds(task, request.first_index, request.second_index)
            except ValueError as exc:
                raise HTTPException(422, str(exc))
            task.recheck = RecheckState(id=uuid4().hex, left_index=left, right_index=right,
                                       start_s=start, end_s=end, processed_s=start)
            saved = save_task(task, request.revision)
            thread = threading.Thread(target=run_recheck, args=(task.id, str(store.root), saved.recheck.id), daemon=True)
            recheck_threads[task.id] = thread
            thread.start()
            return saved

    @app.post('/api/tasks/{task_id}/recheck/cancel', response_model=Task)
    def cancel_recheck(task_id: str, request: RecheckActionRequest):
        task = get_task(task_id)
        if task.recheck is None or task.recheck.id != request.recheck_id:
            raise HTTPException(409, '区域重检已改变，请刷新后重试')
        def cancel(saved):
            saved.recheck.status, saved.recheck.message = 'cancelled', '区域重检已取消，候选证据及原棋谱保留'
        try:
            return store.update_recheck(task_id, request.recheck_id, cancel)
        except RevisionConflict:
            raise HTTPException(409, '重检已经结束，请刷新查看结果')

    @app.post('/api/tasks/{task_id}/recheck/apply', response_model=Task)
    def apply_recheck(task_id: str, request: RecheckActionRequest):
        from .recheck import archive_recheck, integrate_recheck
        task = get_task(task_id)
        job = task.recheck
        if job is None or job.id != request.recheck_id or job.status not in ('ready', 'needs_review', 'applied'):
            raise HTTPException(409, '区域重检尚未结束，请等待检测完成')
        if job.integrated or job.status == 'applied':
            return task
        if task.revision != request.revision:
            raise HTTPException(409, '任务已更新，请刷新后重试')
        archive_recheck(task, store)
        fixed = integrate_recheck(task)
        return save_task(fixed, request.revision)

    return app


app = create_app()
