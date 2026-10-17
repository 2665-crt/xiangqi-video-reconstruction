import hashlib
import json

from .actual import initial_frame_images, verified_initial_candidates
from .calibration import rectify_board, refine_calibration
from .schemas import InitialPieceReview, Position
from .storage import RecoveryCancelled
from .video import iter_frames, save_image
from .vision import VisionRequestError


def run_initial_operation(task, store, vision):
    from .recovery import RecoveryVision, defer_operation
    previous_status = task.recovery.initial_status if task.recovery else task.status
    if task.recovery is not None:
        from .checkpoint import CheckpointStore
        seed = CheckpointStore(store.root).load(task.id, task.recovery.generation, task.recovery.checkpoint)['seed']
        previous_usage = seed['usage']
    else:
        previous_usage = list(task.usage)
    client = RecoveryVision(vision, store, task) if task.recovery else vision
    try:
        task.status, task.stage, task.error = 'analyzing', '核对起始画面与棋子局部图', None
        task.initial_draft, task.initial_piece_reviews = None, []
        task = store.save_analysis(task)
        frames = iter_frames(store.task_dir(task.id) / 'source.mp4', task.start_s, task.end_s)
        try:
            time_s, frame = next(frames)
        finally:
            frames.close()
        if not task.calibration_confirmed:
            task.calibration = refine_calibration(frame, task.calibration)
        image = rectify_board(frame, task.calibration, margin=100)
        images = initial_frame_images(store.task_dir(task.id) / 'source.mp4', time_s, task.calibration, image)
        name = hashlib.sha256(image.tobytes()).hexdigest()[:32] + '.jpg'
        save_image(store.task_dir(task.id) / 'evidence' / name, image)
        task.initial_evidence_ids, task.initial_frame_s = [name], time_s
        task.analysis_stats.model_requests += 1
        task.analysis_stats.images_sent += len(images)
        try:
            task.initial_candidates = client.recognize_position(images, actual=task.replay_mode == 'actual')
        finally:
            task.analysis_stats.model_requests += getattr(client, 'last_request_count', 1) - 1
            task.analysis_stats.images_sent += getattr(client, 'last_image_count', len(images)) - len(images)
        task.initial_candidates = verified_initial_candidates(task.initial_candidates, images)
        diagnostic = getattr(client, 'last_response', None) or {}
        if diagnostic.get('draft_position') is not None:
            task.initial_draft = Position.model_validate(diagnostic['draft_position'])
            task.initial_piece_reviews = [InitialPieceReview.model_validate(r) for r in diagnostic.get('piece_reviews', [])]
        task.stage = '起始画面已识别，请核对并保存初始棋局' if task.initial_candidates else '起始画面无法读清，请调整时间或手动设置棋局'
        if task.initial_draft is not None and not task.initial_candidates:
            task.stage = '候选棋局已显示，请核对标记棋子并保存'
        if not task.initial_candidates:
            task.error = diagnostic.get('error')
        task.status = previous_status if previous_status not in ('analyzing', 'waiting_service', 'action_required') else 'uploaded'
        task.usage = previous_usage + client.usage
        return store.save_analysis(task)
    except RecoveryCancelled:
        return store.get(task.id)
    except VisionRequestError as exc:
        task.usage = previous_usage + client.usage
        task = store.save_analysis(task)
        if task.recovery:
            return defer_operation(store, task, exc)
        raise
    except Exception as exc:
        task.status, task.stage = previous_status, '起始画面识别未完成'
        task.initial_candidates = []
        task.error = str(exc) if isinstance(exc, ValueError) else '无法识别起始画面，请检查视频、棋盘定位和模型连接'
        task.usage = previous_usage + client.usage
        return store.save_analysis(task)
    finally:
        if getattr(client, 'last_response', None):
            (store.task_dir(task.id) / 'vision-last-response.json').write_text(
                json.dumps(client.last_response, ensure_ascii=False, indent=2), encoding='utf-8')
        client.close()
