import importlib.util
from pathlib import Path

from app.schemas import Task, VideoMetadata, opening_position
from app.turns import TurnReplay
from test_move_evidence import moved


def test_regression_gate_rejects_scanned_video_with_missing_tail_moves():
    script = Path(__file__).resolve().parents[2] / 'scripts/check-video-regression.py'
    spec = importlib.util.spec_from_file_location('check_video_regression', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = opening_position()
    baseline = Task(id='b' * 32, filename='baseline.mp4', end_s=10,
                    metadata=VideoMetadata(width=901, height=1001, duration_s=10, codec='h264'),
                    initial_position=before, rules_enabled=True)
    replay = TurnReplay(baseline, before)
    first = moved(before, 'red_rook_0', 0, 8)
    replay.observe(first, 1, [], .5)
    replay.movement(('black',), 7.9)
    replay.observe(moved(first, 'black_rook_0', 0, 1), 9, [], 8)
    replay.end(10)
    baseline.status, baseline.processed_video_s = 'ready', 10
    assert module.check(baseline, baseline)['passed']
    incomplete = baseline.model_copy(deep=True)
    incomplete.moves = incomplete.moves[:1]
    assert not module.check(baseline, incomplete)['passed']
    wrong = baseline.model_copy(deep=True)
    wrong.moves[1].move.target.rank = 2
    assert not module.check(baseline, wrong)['passed']
    wrong_position = baseline.model_copy(deep=True)
    wrong_position.moves[-1].position_after.pieces[0].square.rank = 4
    assert not module.check(baseline, wrong_position)['passed']
    wrong_time = baseline.model_copy(deep=True)
    for record in wrong_time.moves:
        record.action_start_s = record.action_end_s = record.video_start_s = record.video_end_s = 0
    assert not module.check(baseline, wrong_time)['passed']
    wrong_clip = baseline.model_copy(deep=True)
    wrong_clip.moves[1].video_start_s = wrong_clip.moves[1].video_end_s = 0
    assert not module.check(baseline, wrong_clip)['passed']
