import argparse
import json
from pathlib import Path

from app.schemas import Task


def position_state(position):
    if position is None:
        return None
    return (position.side_to_move, sorted((p.id, p.color, p.kind, p.square.file, p.square.rank)
                                          for p in position.pieces))


def check(baseline, result):
    expected = [r.move.model_dump(exclude={'index'}) for r in baseline.moves if r.move is not None]
    observed = [r.move.model_dump(exclude={'index'}) for r in result.moves if r.move is not None]
    issues = []
    if (baseline.start_s, baseline.end_s) != (result.start_s, result.end_s):
        issues.append('分析时间范围发生变化')
    if position_state(baseline.initial_position) != position_state(result.initial_position):
        issues.append('初始棋局发生变化')
    if result.status != 'ready' or any(r.status not in ('auto_verified', 'corrected') for r in result.moves):
        issues.append('仍有未确认步骤或疑点')
    if result.processed_video_s < baseline.end_s:
        issues.append('未扫描到分析区间结束')
    if expected != observed:
        issues.append('走法数量、顺序、棋子身份、起终点或吃子信息与基线不一致')
    if [r.violation for r in baseline.moves if r.move] != [r.violation for r in result.moves if r.move]:
        issues.append('悔棋或其他违规标注与基线不一致')
    states = lambda records: [(position_state(r.position_before), position_state(r.position_after),
                               r.change_type, r.position_edits, r.position_side_to_move) for r in records]
    if states(baseline.moves) != states(result.moves):
        issues.append('逐步局面、轮到方或摆子修正与基线不一致')
    # 历史人工区间复检会改变回放片段边界；动作起落点允许两秒识别偏差。
    times = ('action_start_s', 'action_end_s')
    if len(baseline.moves) == len(result.moves) and any(
            (getattr(a, field) is None) != (getattr(b, field) is None)
            or (getattr(a, field) is not None and getattr(b, field) is not None
                and abs(getattr(a, field) - getattr(b, field)) > 2)
            for a, b in zip(baseline.moves, result.moves) for field in times):
        issues.append('动作起落时间与基线偏差超过两秒')
    previous_landing = result.start_s
    for record in result.moves:
        start, end = record.action_start_s, record.action_end_s
        if (not result.start_s <= record.video_start_s <= record.video_end_s <= result.metadata.duration_s
                or (start is not None and end is not None
                    and not record.video_start_s <= start <= end <= record.video_end_s)
                or (end is not None and end < previous_landing)):
            issues.append('动作或回放片段时间顺序、范围异常')
            break
        if end is not None:
            previous_landing = end
    return {'passed': not issues, 'issues': issues, 'expected_moves': len(expected), 'actual_moves': len(observed),
            'last_move_s': max((r.action_end_s or 0 for r in result.moves if r.move), default=0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('result', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    baseline = Task.model_validate_json(args.baseline.read_text(encoding='utf8'))
    result = Task.model_validate_json(args.result.read_text(encoding='utf8'))
    report = check(baseline, result)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
