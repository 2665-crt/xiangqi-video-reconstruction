import math

import numpy as np
import pytest

from app.fusion import FusionPieceClassifier, fuse_probabilities
from test_piece_classifier import classifier


def test_largest_probability_is_displayed_without_a_confidence_cutoff(classifier, monkeypatch):
    from app.schemas import Piece, Position, Square
    from test_tracking import board_image
    original = Position(pieces=[Piece(id='r',color='red',kind='rook',square=Square(file=4,rank=3))],side_to_move='red')
    monkeypatch.setattr(classifier,'classify',lambda *args: {
        'prediction':'rook','kind':None,'score':.64,'margin':.44,
        'class_probabilities':{'red_rook':.64,'red_elephant':.16,'red_cannon':.20}})
    fusion = FusionPieceClassifier(classifier, detector=None)
    positions = fusion.recognize_position([board_image(original)],actual=True)
    assert positions[0].pieces[0].kind == 'rook'
    assert fusion.last_response['readings'][0]['class_probabilities']['red_rook'] == pytest.approx(.64)
    assert math.isclose(sum(fusion.last_response['readings'][0]['class_probabilities'].values()),1)
    assert positions[0].pieces[0].model_dump().keys() == original.pieces[0].model_dump().keys()


def test_strong_detector_can_help_an_ambiguous_cnn():
    probabilities = fuse_probabilities({'black_elephant':.55,'black_rook':.45},
                                      {'black_rook':.94,'black_elephant':.06},.55,.92)
    assert max(probabilities,key=probabilities.get) == 'black_rook'
    assert sum(probabilities.values()) == pytest.approx(1)


def test_detector_does_not_always_overwrite_a_more_resolved_cnn():
    probabilities = fuse_probabilities({'red_cannon':.64,'red_elephant':.36},
                                      {'red_elephant':.90,'red_cannon':.10},.64,.87)
    assert max(probabilities,key=probabilities.get) == 'red_cannon'


def test_low_confidence_is_a_backend_review_not_an_empty_piece(classifier,monkeypatch):
    from app.schemas import Piece,Position,Square
    from test_tracking import board_image
    original=Position(pieces=[Piece(id='r',color='red',kind='rook',square=Square(file=4,rank=3))],side_to_move='red')
    monkeypatch.setattr(classifier,'classify',lambda *args:{'prediction':'rook','kind':None,'score':.4,'margin':.05,
        'class_probabilities':{'red_rook':.4,'red_cannon':.35,'red_elephant':.25}})
    fusion=FusionPieceClassifier(classifier,detector=None)
    assert fusion.recognize_position([board_image(original)],actual=True)[0].pieces[0].kind == 'rook'
    assert fusion.last_response['piece_reviews'][0]['prediction'] == 'rook'


def test_probabilities_are_computed_from_all_cnn_classes(classifier,monkeypatch):
    from app.piece_classifier import KINDS
    desired=np.array([0,.64,0,.20,0,.16,0],np.float32)
    logits=np.log(np.maximum(desired,1e-20))
    monkeypatch.setattr(classifier.session,'run',lambda *args:[np.tile(logits,(4,1))])
    glyph=np.full((30,30),255,np.uint8)
    glyph[7:23,12:17]=0
    result=classifier.classify(glyph,'red')
    assert len(result['class_probabilities']) == len(KINDS)*2
    assert result['class_probabilities']['red_rook'] == pytest.approx(.64)
    assert result['class_probabilities']['red_elephant'] == pytest.approx(.16)
    assert result['class_probabilities']['red_cannon'] == pytest.approx(.20)


def test_completely_unreadable_glyph_does_not_invent_probabilities(classifier):
    result=classifier.classify(np.full((30,30),255,np.uint8),'red')
    assert result['class_probabilities'] == {}


def test_detector_background_cannot_add_a_piece_to_a_known_empty_square(classifier,monkeypatch):
    from app.schemas import Piece,Position,Square
    from test_tracking import board_image
    original=Position(pieces=[Piece(id='r',color='red',kind='rook',square=Square(file=4,rank=3))],side_to_move='red')
    class Detector:
        def readings(self,image):
            return {(2,2):{'class_probabilities':{'red_rook':1.},'confidence':.99}}
    fusion=FusionPieceClassifier(classifier,Detector())
    positions=fusion.recognize_position([board_image(original)],actual=True)
    assert len(positions[0].pieces)==1
    assert (positions[0].pieces[0].square.file,positions[0].pieces[0].square.rank)==(4,3)


def test_detector_uses_real_class_scores_and_excludes_the_board_class():
    from app.deep_detector import detector_distribution
    names={0:'Red_Rook',1:'Red_Bishop',2:'Red_Cannon',3:'board'}
    probabilities=detector_distribution(np.array([.64,.16,.20,.99]),names)
    assert probabilities['red_rook'] == pytest.approx(.64)
    assert probabilities['red_elephant'] == pytest.approx(.16)
    assert probabilities['red_cannon'] == pytest.approx(.20)
    assert len(probabilities)==14 and sum(probabilities.values())==pytest.approx(1)


def test_local_fusion_is_available_to_frontend_without_cloud_keys(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import create_app
    for name in ('VISION_API_BASE','VISION_MODEL','VISION_API_KEY'):
        monkeypatch.delenv(name,raising=False)
    monkeypatch.setenv('VISION_FUSION','1')
    monkeypatch.setattr('app.fusion.local_fusion_classifier',lambda:object())
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/health').json()['vision_configured'] is True


def test_changed_before_after_frames_produce_a_final_candidate_with_reference(classifier,monkeypatch):
    from app.schemas import Piece,Position,Square
    from test_tracking import board_image
    before=Position(pieces=[Piece(id='r',color='red',kind='rook',square=Square(file=4,rank=3))],side_to_move='red')
    after=before.model_copy(deep=True)
    after.pieces[0].square=Square(file=4,rank=4)
    monkeypatch.setattr(classifier,'classify',lambda *args:{'prediction':'rook','kind':'rook','score':.99,'margin':.9,
        'class_probabilities':{'red_rook':.99,'red_cannon':.01}})
    fusion=FusionPieceClassifier(classifier,None)
    positions=fusion.recognize_position([board_image(before),board_image(after)],reference=before,actual=True)
    assert positions[0].pieces[0].square==Square(file=4,rank=4)


def test_reference_update_does_not_average_the_old_glyph_into_the_final_state(classifier,monkeypatch):
    from app.schemas import Piece,Position,Square
    from test_tracking import board_image
    before=Position(pieces=[Piece(id='r',color='red',kind='rook',square=Square(file=4,rank=3))],side_to_move='red')
    after=before.model_copy(deep=True)
    after.pieces[0].kind='cannon'
    def classify(gray,color):
        if gray[0,0] == 1:
            return {'prediction':'rook','kind':'rook','score':.99,'margin':.98,
                    'class_probabilities':{'red_rook':.99,'red_cannon':.01}}
        return {'prediction':'cannon','kind':'cannon','score':.98,'margin':.96,
                'class_probabilities':{'red_rook':.02,'red_cannon':.98}}
    monkeypatch.setattr('app.fusion.glyph_patch',lambda image,*args: np.full((96,96,3),1 if image is a else 2,np.uint8))
    monkeypatch.setattr('app.fusion.piece_color',lambda *args:'red')
    monkeypatch.setattr(classifier,'classify',classify)
    a,b=board_image(before),board_image(after)
    fusion=FusionPieceClassifier(classifier,None)
    assert fusion.recognize_position([a,b],reference=before,actual=True)[0].pieces[0].kind=='cannon'


def test_colour_conflict_keeps_the_real_detector_distribution(classifier,monkeypatch):
    from app.schemas import Piece,Position,Square
    from test_tracking import board_image
    original=Position(pieces=[Piece(id='r',color='red',kind='horse',square=Square(file=4,rank=3))],side_to_move='red')
    monkeypatch.setattr(classifier,'classify',lambda *args:{'prediction':'horse','kind':None,'score':.55,'margin':.10,
        'class_probabilities':{'red_horse':.55,'red_rook':.45}})
    class Detector:
        def readings(self,image):
            return {(4,3):{'confidence':.91,'class_probabilities':{
                'black_elephant':.99999,'red_rook':.000009,'red_horse':.000001}}}
    fusion=FusionPieceClassifier(classifier,Detector())
    position=fusion.recognize_position([board_image(original)],actual=True)[0]
    assert (position.pieces[0].color,position.pieces[0].kind)==('black','elephant')
    assert fusion.last_response['readings'][0]['class_probabilities']['black_elephant']>.79

def test_deep_readings_recovers_weak_detection_and_keeps_scale_disagreement_unknown():
    from app.deep_detector import DeepPieceDetector
    detector = object.__new__(DeepPieceDetector)
    def reading(label, confidence):
        return {'class_probabilities': {label: .99}, 'confidence': confidence, 'bbox': [0, 0, 30, 30]}
    ordinary = {(0, 0): reading('black_pawn', .69), (1, 0): reading('red_rook', .9)}
    second = {(0, 0): reading('black_pawn', .9), (1, 0): reading('red_cannon', .9)}
    calls = []
    def readings(image, *, image_size=None):
        calls.append(image_size)
        return ordinary if image_size is None else second
    detector.readings = readings
    result = detector.deep_readings(None)
    assert calls == [None, 640]
    assert result[(0, 0)]['confidence'] == .9
    assert (1, 0) not in result
