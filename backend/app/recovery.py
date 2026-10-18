import hashlib
import json
import random
import time
from uuid import uuid4

from .checkpoint import CheckpointStore
from .schemas import Position, RecoveryState, Task
from .storage import RecoveryCancelled
from .vision import VisionRequestError


def prepare_recovery(task, store, operation='analyze', local_only=False):
    generation = uuid4().hex
    seed = task.model_copy(deep=True)
    seed.recovery = None
    checkpoint = CheckpointStore(store.root).save(task.id, generation, {'seed': seed.model_dump(mode='json', by_alias=True)})
    task.recovery = RecoveryState(generation=generation, checkpoint=checkpoint, operation=operation,
                                  lease_token=uuid4().hex, local_only=local_only, initial_status=task.status,
                                  total_requests=task.analysis_stats.model_requests, total_images=task.analysis_stats.images_sent,
                                  initial_requests=task.analysis_stats.model_requests, initial_images=task.analysis_stats.images_sent,
                                  heartbeat_at=time.time())
    return store.save(task, task.revision)


def defer_operation(store, task, error):
    def update(saved):
        r = saved.recovery
        r.attempts += 1
        r.uncertain_attempts += int(error.outcome_uncertain)
        r.last_error, r.phase = error.code, error.phase
        r.waiting_since = r.waiting_since or time.time()
        r.retry_not_before = time.time() + (error.retry_after_s or 0)
        r.lease_token = None
        r.request_started_at = None
        if not error.retryable or r.uncertain_attempts >= 3:
            saved.status, saved.stage = 'action_required', '任务已保存，需要管理员处理视觉服务'
            saved.error = str(error) + ('；重复推理预算已用尽，请管理员检查服务后恢复' if error.retryable else '')
            r.auto_run = False
        else:
            delay = (5, 15, 30, 60, 120)[r.attempts - 1] if r.attempts <= 5 else 300
            delay = min(300, delay * random.uniform(.8, 1.2))
            r.next_attempt_at = max(time.time() + delay, r.retry_not_before)
            saved.status, saved.stage, saved.error = 'waiting_service', '视觉服务暂时不可用，任务和进度已保存，将自动恢复', None
    return store.update_recovery(task.id, task.recovery.generation, task.recovery.lease_token, update)


class RecoveryVision:
    def __init__(self, vision, store, task):
        self.vision, self.store, self.task = vision, store, task
        self.checkpoints = CheckpointStore(store.root)
        self.usage = []
        self.last_response = None
        self.last_request_count, self.last_image_count = 0, 0

    @property
    def configured(self):
        return self.vision.configured

    @property
    def local_classifier(self):
        return getattr(self.vision, 'local_classifier', None)

    def recognize_position(self, images, reference=None, actual=False):
        latest = self.store.get(self.task.id)
        r = latest.recovery
        if (r is None or not r.auto_run or r.generation != self.task.recovery.generation
                or r.lease_token != self.task.recovery.lease_token):
            raise RecoveryCancelled('当前分析已停止或版本已改变')
        parameters = {'version': 1, 'actual': actual, 'reference': reference.model_dump(mode='json') if reference else None,
                      'base_url': getattr(self.vision, 'base_url', ''), 'model': getattr(self.vision, 'model', '')}
        digest = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode())
        for image in images:
            digest.update(str((image.shape, str(image.dtype))).encode())
            digest.update(image.tobytes())
        name = digest.hexdigest() + '.json'
        path = self.checkpoints.directory(latest.id, r.generation) / name
        state = self.checkpoints.load(latest.id, r.generation, name) if path.exists() else dict(parameters, images=images, usage=[])
        if 'positions' in state:
            self.last_request_count = self.last_image_count = 0
            self.last_response = state.get('diagnostic')
            self.usage.extend(state['usage'])
            return [Position.model_validate(p) for p in state['positions']]
        self.checkpoints.save(latest.id, r.generation, state, name)
        local = reference is None and getattr(self.vision, 'local_classifier', None) is not None
        def requested(saved):
            saved.recovery.pending_request = name
            saved.recovery.request_started_at = time.time()
            saved.recovery.heartbeat_at = time.time()
            saved.recovery.total_requests += int(not local)
            saved.recovery.total_images += 0 if local else min(3, len(images))
        self.store.update_recovery(latest.id, r.generation, r.lease_token, requested)
        self.last_request_count, self.last_image_count = int(not local), 0 if local else min(3, len(images))
        before = len(self.vision.usage)
        previous_progress = getattr(self.vision, 'progress', None)
        if hasattr(self.vision, 'progress'):
            def progress(phase):
                def update(saved):
                    saved.recovery.phase = phase
                    saved.recovery.heartbeat_at = time.time()
                self.store.update_recovery(latest.id, r.generation, r.lease_token, update)
            self.vision.progress = progress
        try:
            positions = self.vision.recognize_position(images, reference=reference, actual=actual) if reference is not None else self.vision.recognize_position(images, actual=actual)
        except Exception:
            state['usage'].extend(self.vision.usage[before:])
            state['diagnostic'] = getattr(self.vision, 'last_response', None)
            self.last_response = state['diagnostic']
            self.usage.extend(state['usage'])
            self.checkpoints.save(latest.id, r.generation, state, name)
            if self.last_response:
                (self.store.task_dir(latest.id) / 'vision-last-response.json').write_text(
                    json.dumps(self.last_response, ensure_ascii=False, indent=2), encoding='utf-8')
            raise
        finally:
            if hasattr(self.vision, 'progress'):
                self.vision.progress = previous_progress
        state['usage'].extend(self.vision.usage[before:])
        estimated_requests, estimated_images = self.last_request_count, self.last_image_count
        self.last_request_count = getattr(self.vision, 'last_request_count', estimated_requests)
        self.last_image_count = getattr(self.vision, 'last_image_count', estimated_images)
        state['positions'] = [p.model_dump(mode='json') for p in positions]
        state['diagnostic'] = getattr(self.vision, 'last_response', None)
        self.checkpoints.save(latest.id, r.generation, state, name)
        self.last_response = state['diagnostic']
        self.usage.extend(state['usage'])
        def completed(saved):
            saved.recovery.total_requests += self.last_request_count - estimated_requests
            saved.recovery.total_images += self.last_image_count - estimated_images
            saved.recovery.pending_request = None
            saved.recovery.request_started_at = None
            saved.recovery.heartbeat_at = time.time()
            saved.recovery.attempts = saved.recovery.uncertain_attempts = 0
            saved.recovery.waiting_since = None
            saved.recovery.last_error = saved.recovery.phase = None
            saved.recovery.retry_not_before = 0
        self.store.update_recovery(latest.id, r.generation, r.lease_token, completed)
        return positions

    def close(self):
        self.vision.close()


class ReplayStore:
    def __init__(self, store, current):
        self.store, self.root, self.db = store, store.root, store.db
        self.current = current
        self.frontier = list(current.moves)
        state = CheckpointStore(store.root).load(current.id, current.recovery.generation, current.recovery.checkpoint)
        self.seed = Task.model_validate(state['seed'])
        self.target_video_s = current.processed_video_s
        self.target_event_s = max(current.recognition_video_s,
            max((record.video_end_s for record in current.moves), default=current.start_s),
            current.pending_turn.video_end_s if current.pending_turn else current.start_s)
        self.caught_up = (len(self.seed.moves) >= len(self.frontier)
                          and self.seed.processed_video_s >= self.target_video_s
                          and self.seed.recognition_video_s >= self.target_event_s)
        self.seed.recovery = current.recovery.model_copy(deep=True)
        self.seed.status, self.seed.stage = 'analyzing', '正在重建已保存的检测状态'

    def get(self, task_id):
        return self.seed.model_copy(deep=True)

    def task_dir(self, task_id):
        return self.store.task_dir(task_id)

    def save(self, task, expected_revision):
        return self.save_analysis(task)

    def save_analysis(self, task):
        latest = self.store.get(task.id)
        if (latest.recovery is None or not latest.recovery.auto_run
                or latest.recovery.generation != task.recovery.generation
                or latest.recovery.lease_token != task.recovery.lease_token):
            raise RecoveryCancelled('当前分析已停止或版本已改变')
        if not self.caught_up and (len(task.moves) < len(self.frontier)
                                  or task.processed_video_s < self.target_video_s
                                  or task.recognition_video_s < self.target_event_s):
            if task.status not in ('analyzing', 'waiting_service'):
                raise ValueError('恢复重建没有覆盖已保存棋谱，原棋谱已保留，请管理员检查')
            def rebuilding(saved):
                saved.rebuild_video_s = task.processed_video_s
                saved.rebuild_target_s = latest.processed_video_s
                saved.stage = '正在重建安全检查点之后的历史局面'
                saved.recovery.heartbeat_at = time.time()
                saved.scan_progress_s = max(saved.scan_progress_s, task.scan_progress_s)
                saved.scan_complete = saved.scan_complete or task.scan_complete
                saved.scan_events = max(saved.scan_events, task.scan_events)
                saved.pending_event_count = max(saved.pending_event_count, task.pending_event_count)
            latest = self.store.update_recovery(task.id, task.recovery.generation, task.recovery.lease_token, rebuilding)
            task.revision, task.recovery = latest.revision, latest.recovery.model_copy(deep=True)
            return task
        fields = ('move', 'video_start_s', 'video_end_s', 'position_after', 'observed_positions', 'violation', 'change_type')
        if not self.caught_up and any(any(getattr(old, key) != getattr(new, key) for key in fields)
               for old, new in zip(self.frontier, task.moves)):
            raise ValueError('恢复重建与已保存棋谱不一致，原棋谱已保留，请管理员检查')
        copy = task.model_copy(update={'moves': list(task.moves)})
        copy.rebuild_video_s = copy.rebuild_target_s = None
        copy.processed_video_s = max(copy.processed_video_s, latest.processed_video_s)
        if not self.caught_up:
            for i, record in enumerate(self.frontier):
                copy.moves[i] = record.model_copy(deep=True)
            self.caught_up = True
        saved = self.store.save_analysis(copy)
        task.revision, task.recovery = saved.revision, saved.recovery
        return task

    def save_progress(self, task):
        if not self.caught_up:
            return self.save_analysis(task)
        saved = self.store.save_progress(task)
        task.revision, task.recovery = saved.revision, saved.recovery
        return task


def save_boundary_checkpoint(store, task, time_s, context, tracking_calibration=None, motion_active=False, motion_reference_s=None):
    backing = getattr(store, 'store', store)
    if task.recovery is None or not task.recovery.auto_run:
        return
    latest = backing.get(task.id)
    if len(task.moves) < len(latest.moves):
        return
    seed = task.model_copy(deep=True)
    seed.context_position = context.model_copy(deep=True)
    seed.pending_turn = None
    seed.resume_video_s = time_s
    seed.resume_calibration = tracking_calibration
    seed.resume_motion_active = motion_active
    seed.resume_motion_reference_s = motion_reference_s
    seed.processed_video_s = time_s
    seed.recognition_video_s = time_s
    seed.recovery = None
    name = CheckpointStore(backing.root).save(task.id, task.recovery.generation,
                                            {'seed': seed.model_dump(mode='json', by_alias=True)})
    def update(saved):
        saved.recovery.checkpoint = name
        saved.recovery.heartbeat_at = time.time()
    backing.update_recovery(task.id, task.recovery.generation, task.recovery.lease_token, update)


def run_saved_operation(task_id, store, vision=None):
    from .analysis import analyze_task
    from .vision import VisionClient
    task = store.get(task_id)
    try:
        if task.recovery.operation == 'initial':
            from .initial import run_initial_operation
            return run_initial_operation(task, store, vision or VisionClient())
        replay = ReplayStore(store, task)
        client = None if task.recovery.local_only else RecoveryVision(vision or VisionClient(), store, task)
        analyze_task(task_id, replay, client)
    except RecoveryCancelled:
        return
    except Exception as exc:
        def failed(saved):
            saved.status, saved.stage = 'action_required', '恢复状态无法安全应用，请管理员检查'
            saved.error = str(exc) if isinstance(exc, ValueError) else '无法读取恢复数据，请检查任务文件和磁盘空间'
            saved.recovery.auto_run = False
            saved.recovery.lease_token = None
        try:
            store.update_recovery(task.id, task.recovery.generation, task.recovery.lease_token, failed)
        except RecoveryCancelled:
            pass


from .recovery_scheduler import RecoveryCoordinator
