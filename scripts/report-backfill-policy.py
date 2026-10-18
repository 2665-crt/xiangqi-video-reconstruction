import difflib
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'output/backfill-policy-20261016'
    before = json.loads((output / 'fusion-v2-before/summary.json').read_text(encoding='utf8'))
    after = json.loads((output / 'fusion-v2-after/summary.json').read_text(encoding='utf8'))
    api = json.loads((output / 'api-validation.json').read_text(encoding='utf8'))
    assert before['range_s'] == after['range_s'] == [47, 192.1]
    assert before['local_fusion_fallback'] and after['local_fusion_fallback']
    assert before['error'] is None and after['error'] is None
    assert before['moves'] == after['moves'], 'Move/status/timing/violation sequence changed'
    assert before['confirmed_prefix_length'] == after['confirmed_prefix_length'] == 14
    assert after['backfill_intervals'] == 1 and after['backfill_frames'] == 47
    assert api['status'] == 'ready' and api['first_seven_match_ground_truth']
    summary = {'range_s': after['range_s'], 'source_task_id': after['source_task_id'],
        'before': {k: before[k] for k in ('backfill_intervals', 'backfill_frames', 'backfill_seconds')},
        'after': {k: after[k] for k in ('backfill_intervals', 'backfill_frames', 'backfill_seconds')},
        'backfill_frame_reduction_percent': round(100 * (1 - after['backfill_frames'] / before['backfill_frames']), 2),
        'move_status_timing_violation_sequences_equal': True, 'confirmed_moves': 14,
        'unresolved_records': sum(r['status'] == 'needs_review' for r in after['moves']),
        'backend_tests': {'passed': 482, 'skipped': 2, 'warnings': 1},
        'frontend_tests_passed': 57, 'frontend_build_passed': True,
        'api_validation': api, 'full_video_accuracy_claimed': False,
        'timing_note': 'Fusion comparison processes partly overlapped. Frame and interval counts are the primary performance evidence; wall times are not a controlled benchmark.'}
    changes = [('backend/app/actual.py', 'actual.py'), ('backend/app/board_evidence.py', 'board_evidence.py'),
               ('backend/app/schemas.py', 'schemas.py'), ('frontend/src/App.tsx', 'App.tsx'), ('frontend/src/types.ts', 'types.ts'),
               ('backend/tests/test_backfill_policy.py', None), ('scripts/validate-backfill-policy.py', None),
               ('scripts/report-backfill-policy.py', None)]
    patch = []
    hashes = {}
    for relative, backup in changes:
        original = (output / 'before' / backup).read_text(encoding='utf8').splitlines(keepends=True) if backup else []
        current = (root / relative).read_text(encoding='utf8').splitlines(keepends=True)
        patch.extend(difflib.unified_diff(original, current, fromfile=f'before/{relative}', tofile=f'after/{relative}'))
        hashes[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    (output / 'changes.patch').write_text(''.join(patch), encoding='utf8')
    summary['source_sha256'] = hashes
    (output / 'validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    report = f'''# 减少重复回查验证

已更新当前融合识别实验程序；服务地址 http://127.0.0.1:5290，后端 8290。

## 本次策略

- 起终点证据不足、无变化噪声、单步之外的占位误识别：继续向后取证，不立即倒回视频。
- 仅在唯一候选明确包含无法用单步解释的变化、相关落点可见且没有可见占位矛盾时启动自动回查。
- 按上一确认边界合并不同候选造成的同一疑点；本区间最多回查一次，跨度超过 15 秒保留待核对。
- 冲突仍不能推进可信局面；原有悔棋识别、证据和人工区间重检保留。
- 页面显示本次局部回查次数和耗时，未记录过此统计的旧任务不显示虚假的零次。

## 截图对应录屏实际验证

源任务 `{after['source_task_id']}`，原视频 47～192.1 秒；新旧均启用本地融合模型补充候选。

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| 自动回查次数 | {before['backfill_intervals']} | {after['backfill_intervals']} |
| 回查扫描帧数 | {before['backfill_frames']} | {after['backfill_frames']} |
| 已确认走法 | 14 | 14 |
| 待核对记录 | {summary['unresolved_records']} | {summary['unresolved_records']} |

回查扫描帧数减少 {summary['backfill_frame_reduction_percent']}%。走法、状态、落点时间和违规标注序列完全一致。疑点仍保留，没有凭猜测补步。

新旧进程部分时段同时运行，因此耗时仅保留在原始 JSON，不作为严格速度基准。本次未重新验证完整 33 分钟视频的准确率。

## 悔棋 API 和页面验证

新增独立任务 `{api['task_id']}`，原视频 18～58 秒，HTTP 健康检查 200、启动 202、结果读取 200。
最终 `{api['status']}`，8 条自动确认操作，回查 0 次。前 7 条与此前人工逐帧核对的基准一致：黑马返回原位、黑方改走炮、随后红马八进七。悔棋的原移动与返回记录均标注实际回退方为黑方。没有人工覆盖步骤。

页面截图：`ui-backfill-count.png`。

## 自动验证

- 后端：482 passed，2 skipped，1 条既有 Starlette/AnyIO 弃用警告。
- 前端：57 passed，生产构建通过。
- 新增 9 个回查策略/兼容测试；其中前 6 个在修改前复现失败，旧统计兼容测试也先复现失败后修复。
- 独立只读代码审查未发现实质问题。

完整本次源码差异见 `changes.patch`；原始源代码快照在 `before/`，原始对照结果在 `fusion-v2-before/` 和 `fusion-v2-after/`，API 结果在 `api-task.json`。
'''
    (output / '验证报告.md').write_text(report, encoding='utf8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('api_validation', 'source_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
