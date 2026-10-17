import logging
import os
import shutil
import threading
import time

from .storage import RecoveryCancelled, RevisionConflict
from .vision import VisionRequestError


class RecoveryCoordinator:
    def __init__(self, store, workers, analysis_lock, launch):
        self.store, self.workers, self.analysis_lock, self.launch = store, workers, analysis_lock, launch
        self.stopped = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopped.wait(2):
            try:
                self.tick()
            except Exception:
                logging.getLogger(__name__).exception('后台恢复调度未完成，将在下一轮检查')

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=5)

    def tick(self, now=None):
        from .recovery import defer_operation
        if not self.analysis_lock.acquire(blocking=False):
            return
        try:
            for task_id, process in list(self.workers.items()):
                if process.is_alive():
                    task = self.store.get(task_id)
                    r = task.recovery
                    clock = time.time() if now is None else now
                    if r is None or not r.auto_run:
                        return
                    deadline = 360 if r.request_started_at else 120
                    last_activity = r.request_started_at or r.heartbeat_at
                    if not last_activity or clock - last_activity <= deadline:
                        return
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=2)
                    self.workers.pop(task_id, None)
                    defer_operation(self.store, task, VisionRequestError(
                        '分析进程长时间没有推进，已保留状态并准备恢复', 'WorkerStalled', 'worker',
                        outcome_uncertain=r.request_started_at is not None))
                    continue
                process.join(timeout=0)
                self.workers.pop(task_id, None)
                task = self.store.get(task_id)
                if task.status == 'analyzing' and task.recovery and task.recovery.auto_run:
                    defer_operation(self.store, task, VisionRequestError('分析进程意外退出，保存的任务将自动恢复',
                        'WorkerExited', 'worker', outcome_uncertain=task.recovery.request_started_at is not None))
            for task in self.store.list_tasks():
                if task.status != 'waiting_service' or task.recovery is None:
                    continue
                claimed = self.store.claim_recovery(task.id, task.recovery.generation, time.time() if now is None else now)
                if claimed is None:
                    continue
                try:
                    self.workers[task.id] = self.launch(claimed)
                except Exception:
                    defer_operation(self.store, claimed, VisionRequestError('无法启动恢复进程，将自动重试', 'WorkerStartFailed', 'worker'))
                break
        finally:
            self.analysis_lock.release()

    def request_resume(self, task_id, revision):
        from .recovery import prepare_recovery
        with self.analysis_lock:
            task = self.store.get(task_id)
            if task.revision != revision:
                raise RevisionConflict('任务已更新，请重新读取后恢复')
            if task.recovery is None:
                if task.status not in ('failed', 'interrupted'):
                    raise ValueError('当前任务没有需要恢复的操作')
                for existing in self.store.list_tasks():
                    if (existing.recovery is not None and existing.recovery.origin_task_id == task.id
                            and existing.recovery.origin_revision == task.revision):
                        return existing
                origin_id, origin_revision = task.id, task.revision
                source = self.store.task_dir(task.id) / 'source.mp4'
                if not source.exists():
                    raise ValueError('原视频不存在，无法重建恢复任务；已有棋谱已保留')
                restored = self.store.create(task.filename + '（恢复副本）', task.metadata)
                for field in ('start_s', 'end_s', 'calibration', 'calibration_confirmed', 'initial_position', 'context_position',
                              'initial_evidence_ids', 'initial_frame_s', 'initial_turn_known', 'replay_mode', 'rules_enabled'):
                    setattr(restored, field, getattr(task, field))
                target = self.store.task_dir(restored.id)
                try:
                    os.link(source, target / 'source.mp4')
                except OSError:
                    shutil.copy2(source, target / 'source.mp4')
                (target / 'evidence').mkdir(exist_ok=True)
                for name in task.initial_evidence_ids:
                    evidence = self.store.task_dir(task.id) / 'evidence' / name
                    if evidence.exists():
                        shutil.copy2(evidence, target / 'evidence' / name)
                task = prepare_recovery(restored, self.store)
                task.recovery.origin_task_id, task.recovery.origin_revision = origin_id, origin_revision
            elif task.status == 'analyzing' and task.recovery.auto_run:
                return task
            elif task.recovery.checkpoint is None:
                raise ValueError('恢复检查点已失效，请重新开始分析；已有棋谱已保留')
            elif task.status not in ('waiting_service', 'action_required', 'interrupted', 'failed'):
                raise ValueError('当前任务无需恢复')
            task.status, task.stage, task.error = 'waiting_service', '恢复操作已排队，后台将自动处理', None
            r = task.recovery
            r.auto_run, r.lease_token = True, None
            r.uncertain_attempts = 0
            r.next_attempt_at = max(time.time(), r.retry_not_before)
            return self.store.save(task, task.revision)
