import numpy as np

from app.schemas import Piece, Position, Square, opening_position


def test_partial_observation_does_not_search_history_that_cannot_match():
    from app.actual import _cells_position
    class UnreadableHistory:
        @property
        def pieces(self):
            raise AssertionError('partial observation searched history')
    assert _cells_position(opening_position(),{(4,4):'red'},[UnreadableHistory()]) is None


def test_indexed_history_picks_latest_identical_board_without_aliasing():
    from app.actual import PositionHistory, reconcile_position
    old = Position(side_to_move='red',pieces=[Piece(id='old',color='red',kind='rook',square=Square(file=0,rank=0))])
    latest = old.model_copy(deep=True)
    latest.pieces[0].id = 'latest'
    history = PositionHistory([old,latest])
    observed = old.model_copy(deep=True)
    result = reconcile_position(old,observed,history)
    assert result.pieces[0].id == 'latest'
    result.pieces[0].square = Square(file=1,rank=0)
    assert latest.pieces[0].square == Square(file=0,rank=0)


def test_history_catalog_keeps_captured_id_and_latest_piece_information():
    from app.actual import PositionHistory,reconcile_position
    old = Position(side_to_move='red',pieces=[Piece(id='r',color='red',kind='rook',square=Square(file=0,rank=0))])
    current = Position(side_to_move='black',pieces=[])
    observed = Position(side_to_move='red',pieces=[Piece(id='unknown',color='red',kind='rook',square=Square(file=1,rank=0))])
    indexed = reconcile_position(current,observed,PositionHistory([old,current]))
    reference = reconcile_position(current,observed,[old,current])
    assert indexed == reference
    assert indexed.pieces[0].id == 'r'


def test_complete_ambiguous_observation_matches_latest_occupancy():
    from app.actual import PositionHistory,_cells_position
    initial=opening_position()
    older=initial.model_copy(deep=True)
    newer=initial.model_copy(deep=True)
    for p in older.pieces:
        if p.id=='red_rook_0':p.square=Square(file=0,rank=8)
        if p.id=='red_rook_8':p.square=Square(file=8,rank=8)
    newer=older.model_copy(deep=True)
    newer.side_to_move='black'
    cells={(f,r):None for f in range(9) for r in range(10)}
    cells.update({(p.square.file,p.square.rank):p.color for p in newer.pieces})
    expected=_cells_position(initial,cells,[initial,older,newer])
    actual=_cells_position(initial,cells,PositionHistory([initial,older,newer]))
    assert actual==expected==newer


def test_scan_watermark_count_handles_unsorted_duplicate_event_times(tmp_path,video_path):
    import shutil
    from app.scanning import VideoScan
    from app.schemas import Calibration,VideoMetadata
    from app.storage import TaskStore
    store=TaskStore(tmp_path)
    task=store.create('index.mp4',VideoMetadata(width=128,height=128,duration_s=10,codec='h264'))
    shutil.copy2(video_path,store.task_dir(task.id)/'source.mp4')
    task.calibration=Calibration(corners=[{'x':0,'y':0},{'x':127,'y':0},{'x':127,'y':127},{'x':0,'y':127}])
    scan=VideoScan(task,store)
    from app.analysis import VideoEvent
    for i,t in enumerate([3.,1.,3.,5.]):
        scan._append(VideoEvent(t,[np.full((3,3,3),i,np.uint8)]))
    class NoIteration(list):
        def __iter__(self):
            raise AssertionError('progress update walked all historical events')
    scan.index['events']=NoIteration(scan.index['events'])
    for watermark in [0.,3.,5.,1.5]:
        task.recognition_video_s=watermark
        scan.update(task)
        assert task.pending_event_count==sum(t>watermark for t in [3.,1.,3.,5.])


def test_active_progress_read_does_not_replay_all_saved_moves(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from test_manual_replay import fixture_task
    app=create_app(tmp_path)
    calls=[]
    monkeypatch.setattr('app.turns.revalidate_turns',lambda task:(calls.append(task.id) or task))
    with TestClient(app) as client:
        task=fixture_task(app.state.store)
        for status in ('analyzing','waiting_service'):
            task.status=status
            task=app.state.store.save(task,task.revision)
            response=client.get(f'/api/tasks/{task.id}')
            assert response.status_code==200
            assert response.json()['moves']==task.model_dump(mode='json',by_alias=True)['moves']
        assert calls==[]
        task.status='ready'
        task=app.state.store.save(task,task.revision)
        client.get(f'/api/tasks/{task.id}')
        assert calls==[task.id]
