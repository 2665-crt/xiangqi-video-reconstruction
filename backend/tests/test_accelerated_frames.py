import time

import numpy as np


def test_compiled_circle_support_matches_original_numpy_rounding_and_border_clip():
    from app.numerical import circle_support
    rng=np.random.default_rng(21)
    edges=(rng.integers(0,20,(110,160),dtype=np.uint8)==0).astype(np.uint8)*255
    angles=np.linspace(0,2*np.pi,96,endpoint=False)
    for _ in range(100):
        cx,cy,radius=np.float32(rng.uniform(-10,170)),np.float32(rng.uniform(-10,120)),np.float32(rng.uniform(13,27))
        support=np.zeros(len(angles),bool)
        for delta in (-2,-1,0,1,2):
            xs=np.clip(np.rint(cx+(radius+delta)*np.cos(angles)).astype(int),0,edges.shape[1]-1)
            ys=np.clip(np.rint(cy+(radius+delta)*np.sin(angles)).astype(int),0,edges.shape[0]-1)
            support|=edges[ys,xs]>0
        assert circle_support(edges,cx,cy,radius)==float(support.mean())


def test_compiled_finger_counts_match_original_float64_mask():
    from app.numerical import finger_counts
    rng=np.random.default_rng(14)
    small=rng.integers(0,256,(560,510,3),dtype=np.uint8)
    hsv=rng.integers(0,256,small.shape,dtype=np.uint8)
    squares=np.array([(f,r) for r in range(10) for f in range(9)],dtype=np.int64)
    dy,dx=np.mgrid[-29:30,-29:30]
    ring=(dx*dx+dy*dy>=22*22)&(dx*dx+dy*dy<29*29)
    expected=[]
    for f,r in squares:
        x,y=55+f*50,55+r*50
        pixels=small[y-29:y+30,x-29:x+30][ring].astype(float)
        colors=hsv[y-29:y+30,x-29:x+30][ring]
        mask=((colors[:,0]<23)&(colors[:,1]>55)&(colors[:,2]>95)
              &(pixels[:,2]>pixels[:,1]*1.2)&(pixels[:,0]>pixels[:,1]*.62))
        expected.append(int(mask.sum()))
    assert finger_counts(small,hsv,squares).tolist()==expected


def test_parallel_batches_keep_source_order_bound_prefetch_and_propagate_failure():
    from app.preprocessing import parallel_batches
    consumed=[]
    def source():
        for i in range(30):
            consumed.append(i)
            yield i
    def process(i):
        time.sleep(.002*(3-i%3))
        return i*2
    with parallel_batches(source(),process,workers=3,batch_size=6) as results:
        assert next(results)==0
        assert len(consumed)==6
        assert [next(results) for _ in range(5)]==[2,4,6,8,10]
        assert len(consumed)==6
        assert list(results)==list(range(12,60,2))
    def fail(i):
        if i==2:raise ValueError('broken frame')
        return i
    import pytest
    with pytest.raises(ValueError,match='broken frame'):
        with parallel_batches(iter(range(9)),fail,workers=2,batch_size=4) as results:
            assert next(results)==0
            assert next(results)==1
            next(results)


def test_parallel_batches_deliver_valid_frames_before_decoder_failure():
    import pytest
    from app.preprocessing import parallel_batches
    def source():
        yield 1
        yield 2
        raise ValueError('decode failure')
    with parallel_batches(source(),lambda frame:frame*2) as results:
        assert next(results)==2
        assert next(results)==4
        with pytest.raises(ValueError,match='decode failure'):
            next(results)


def test_prepared_observation_uses_current_reference_and_returns_independent_dict(monkeypatch):
    from collections import OrderedDict
    import weakref
    import app.actual as actual
    from app.schemas import Position,Piece,Square
    image=np.zeros((110,100,3),np.uint8)
    image.setflags(write=False)
    raw={(f,r):None for f in range(9) for r in range(10)}
    enhanced={(4,4):'red'}
    cache=OrderedDict({(id(image),'prepared'):(weakref.ref(image),(raw,raw.copy(),enhanced))})
    token=actual._tracking_cache.set(cache)
    try:
        empty=Position(pieces=[],side_to_move='red')
        changed=Position(pieces=[Piece(id='r',color='red',kind='rook',square=Square(file=4,rank=4))],side_to_move='black')
        assert actual._tracking_cells(image,empty)==raw
        result=actual._tracking_cells(image,changed)
        assert result==enhanced
        result.clear()
        assert actual._tracking_cells(image,changed)==enhanced
    finally:actual._tracking_cache.reset(token)


def test_missing_prefetched_enhancement_is_computed_only_for_current_reference(monkeypatch):
    from collections import OrderedDict
    import weakref
    import app.actual as actual
    from app.schemas import Position,Piece,Square
    image=np.zeros((110,100,3),np.uint8)
    image.setflags(write=False)
    raw={(f,r):None for f in range(9) for r in range(10)}
    cache=OrderedDict({(id(image),'prepared'):(weakref.ref(image),(raw,raw.copy(),None))})
    calls=[]
    monkeypatch.setattr(actual,'_finish_tracking_cells',lambda *args:(calls.append(args[2]) or {(4,4):'red'}))
    token=actual._tracking_cache.set(cache)
    try:
        empty=Position(pieces=[],side_to_move='red')
        changed=Position(pieces=[Piece(id='r',color='red',kind='rook',square=Square(file=4,rank=4))],side_to_move='black')
        assert actual._tracking_cells(image,empty)==raw
        assert calls==[]
        assert actual._tracking_cells(image,changed)=={(4,4):'red'}
        assert actual._tracking_cells(image,changed)=={(4,4):'red'}
        assert calls==[True]
    finally:actual._tracking_cache.reset(token)


def test_preprocessing_skips_expensive_enhancement_for_unchanged_complete_board(monkeypatch):
    import app.actual as actual
    raw={(f,r):None for f in range(9) for r in range(10)}
    image=np.zeros((110,100,3),np.uint8)
    monkeypatch.setattr(actual,'rectify_board',lambda *a,**kw:image)
    monkeypatch.setattr('app.analysis.read_board_cells',lambda *a,**kw:raw.copy())
    calls=[]
    monkeypatch.setattr(actual,'_finish_tracking_cells',lambda image,cells,expand,reference_cells=None:(calls.append(expand) or cells.copy()))
    result=actual._prepare_tracking_frame((1.,image,0.,{}),None,None,2,False)
    assert calls==[False]
    assert result[3][2] is None
    calls.clear()
    result=actual._prepare_tracking_frame((1.,image,0.,{(4,4):'red'}),None,None,2,False)
    assert calls==[True,False]
    assert result[3][2]==raw
