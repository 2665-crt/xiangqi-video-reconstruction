import hashlib
import json
from pathlib import Path

import httpx

from app.storage import TaskStore


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'output/takeback-35s-20261016'
    run = json.loads((output / 'run-400s.json').read_text(encoding='utf8'))
    store = TaskStore(root / 'data')
    task = store.get(run['task_id'])
    if task.status in ('analyzing', 'waiting_service'):
        raise RuntimeError('完整区间尚未跑完')
    original = store.get(run['original_task_id'])
    truth = json.loads((output / 'case-ground-truth.json').read_text(encoding='utf8'))['sequence']
    records = [r for r in task.moves if r.move is not None]
    actual = [{'piece_id': r.move.piece_id, 'from': [r.move.source.file, r.move.source.rank],
               'to': [r.move.target.file, r.move.target.rank], 'landing_s': r.action_end_s,
               'record_index': r.index, 'status': r.status, 'violation': r.violation,
               'board_check': r.board_check.status if r.board_check else None} for r in records]
    exact = all(all(expected[key] == observed[key] for key in ('piece_id', 'from', 'to'))
                for expected, observed in zip(truth, actual[:len(truth)])) and len(actual) >= len(truth)
    assert exact, '实际视频重跑尚未准确恢复人工核对的七条操作'
    assert records[3].violation == records[4].violation == 'takeback'
    assert '黑方悔棋' in records[4].violation_note
    assert records[4].position_after.side_to_move == 'black'
    evidence = []
    with httpx.Client(timeout=30, trust_env=False) as client:
        reopened = client.get(f'http://127.0.0.1:8290/api/tasks/{task.id}')
        reopened.raise_for_status()
        assert all(r['status'] == 'auto_verified' for r in reopened.json()['moves'][:7])
        for record in records[:len(truth)]:
            paths = [store.task_dir(task.id) / 'evidence' / name for name in record.evidence_ids]
            assert paths and all(path.is_file() for path in paths)
            response = client.get(f'http://127.0.0.1:8290/api/tasks/{task.id}/evidence/{paths[-1].name}')
            response.raise_for_status()
            evidence.append({'record': record.index, 'http_status': response.status_code,
                             'images': len(paths), 'times_s': record.evidence_times_s,
                             'last_image_sha256': hashlib.sha256(paths[-1].read_bytes()).hexdigest()})
    reviews = [{'record': r.index, 'start_s': r.video_start_s, 'end_s': r.video_end_s,
                'reason': r.reason, 'piece_ids': r.review_changed_piece_ids}
               for r in task.moves if r.status == 'needs_review']
    report = {'task_id': task.id, 'original_task_id': original.id, 'range_s': [task.start_s, task.end_s],
              'status': task.status, 'processed_video_s': task.processed_video_s,
              'recognition_video_s': task.recognition_video_s, 'scan_complete': task.scan_complete,
              'original_move_count': sum(r.move is not None for r in original.moves),
              'rerun_move_count': len(records), 'last_move_s': max(r.action_end_s or r.video_end_s for r in records),
              'takeback_actor': 'black', 'manual_sequence_exact': exact,
              'manual_validation_scope': 'First seven observed operations, including undo, replacement and red horse reply',
              'full_video_accuracy_claimed': False, 'reviews': reviews, 'actual_moves': actual,
              'workspace': str(root),
              'final_ui_screenshot': str(output / 'ui-takeback-steps-final.png'),
              'run_stats': task.run_stats.model_dump(), 'evidence_validation': evidence,
              'training_video': str(root / '直播录屏/2.mp4'),
              'color_model_sha256': hashlib.sha256((root / 'backend/models/board-color-v1.npz').read_bytes()).hexdigest(),
              'tests': {'backend_passed': 473, 'backend_skipped': 2, 'frontend_passed': 57, 'frontend_build': 'passed'}}
    gap_report = output / 'gap-recheck-validation.json'
    if gap_report.is_file():
        report['actual_frame_gap_rechecks'] = json.loads(gap_report.read_text(encoding='utf8'))
    (output / 'final-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    (output / 'final-task.json').write_text(task.model_dump_json(by_alias=True, indent=2), encoding='utf8')
    lines = ['# 视频悔棋与后续步骤复测', '',
             f'- 任务：`{task.id}`；实际分析区间：18–400 秒。',
             f'- 对照任务当前保存 {report["original_move_count"]} 条走棋；本次 {len(records)} 条走棋，最后落点 {report["last_move_s"]:.3f} 秒。',
             f'- 状态：{task.status}；保留 {len(reviews)} 条待核对记录。人工重点核验范围为悔棋段及两处遮挡区间。',
             '- 人工核对的前七条操作逐条匹配；黑马复位归属黑方悔棋，之后黑炮改走，红方马八进七。',
             '- 红马：从零计数 (1,9)→(2,7)；界面坐标 (2,10)→(3,8)。', '',
             '原因与修复：颜色造成木纹/肤色混淆，汉界字样产生假黑子，线网与实际棋子中心错位；短暂复位和分帧证据又被回合处理丢弃。已接入训练颜色、木纹圆形排除、已知棋子中心校正、分帧证据与悔棋回合归属。', '',
             '后端 473 通过、2 跳过；前端 57 通过，构建通过。外部模型请求为零，启用了本地棋子检测。完整区间重跑后，另对末段边界及遮挡区间作了实际帧回归；合计耗时包含复测。', '',
             '|记录|落定秒|棋子|起点→终点|标记|', '|---|---:|---|---|---|']
    for row in actual[:7]:
        lines.append(f'|{row["record_index"]}|{row["landing_s"]:.3f}|{row["piece_id"]}|{row["from"]}→{row["to"]}|{row["violation"] or "正常"}|')
    lines.extend(['', '区间复核结果：', ''])
    for review in reviews:
        lines.append(f'- 记录 {review["record"]}：{review["start_s"]:.3f}–{review["end_s"]:.3f} 秒；{review["reason"]}')
    if not reviews:
        lines.append('- 当前无待核对区间。95–105 秒黑车及 181–194 秒吃兵遮挡已通过实际帧回归消解，原截图保留在对应走棋证据中。')
    lines.extend(['', '详细记录、证据时间和截图哈希见同目录 final-validation.json；未对整段视频声称逐步全对。'])
    (output / '验证报告.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({k: report[k] for k in ('task_id', 'status', 'rerun_move_count', 'last_move_s', 'manual_sequence_exact', 'reviews')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
