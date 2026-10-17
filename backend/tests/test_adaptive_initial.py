import cv2
import numpy as np
import pytest

from app.schemas import Piece, Position, Square
from test_piece_classifier import classifier


def physical_board(pieces, background=(100, 175, 200), face=(225, 225, 225), offset=-24):
    image = np.full((1001, 901, 3), background, np.uint8)
    for file, rank, color in pieces:
        center = (50 + file * 100, 50 + rank * 100 + offset)
        cv2.circle(image, center, 41, (30, 30, 30), 3)
        cv2.circle(image, center, 38, face, -1)
        cv2.putText(image, 'P', (center[0] - 15, center[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (50, 30, 180) if color == 'red' else (30, 30, 30), 5)
    return image


@pytest.mark.parametrize('background,face', [((100, 175, 200), (225, 225, 225)),
    ((100, 80, 45), (225, 225, 225)), ((85, 135, 175), (105, 170, 220))])
def test_visible_pieces_above_grid_are_retained_at_arbitrary_squares(classifier, background, face):
    image = physical_board([(2, 4, 'red'), (7, 5, 'black')], background, face)
    classifier.recognize_position([image], actual=True)
    draft = classifier.last_response['draft_position']
    assert {(p['square']['file'], p['square']['rank']) for p in draft['pieces']} == {(2, 4), (7, 5)}


def test_complementary_occlusions_keep_pieces_from_both_frames(classifier):
    images = [physical_board([(2, 4, 'red'), (7, 5, 'black')], offset=0) for _ in range(2)]
    cv2.rectangle(images[0], (700, 500), (800, 600), (90, 130, 190), -1)
    cv2.rectangle(images[1], (200, 400), (300, 500), (90, 130, 190), -1)
    classifier.recognize_position(images, actual=True)
    pieces = classifier.last_response['draft_position']['pieces']
    assert {(p['square']['file'], p['square']['rank']) for p in pieces} == {(2, 4), (7, 5)}


def test_multiframe_completion_is_not_silently_confirmed(classifier):
    images = [physical_board([(2, 4, 'red'), (7, 5, 'black')], offset=0) for _ in range(2)]
    cv2.rectangle(images[0], (700, 500), (800, 600), (90, 130, 190), -1)
    cv2.rectangle(images[1], (200, 400), (300, 500), (90, 130, 190), -1)
    assert classifier.recognize_position(images, actual=True) == []
    assert any(r['square'] == {'file': 2, 'rank': 4} for r in classifier.last_response['piece_reviews'])


def test_complete_occlusion_without_glyph_does_not_become_an_empty_board(classifier):
    image = np.full((1001, 901, 3), (90, 130, 190), np.uint8)
    assert classifier.recognize_position([image], actual=True) == []
    assert not classifier.last_response['draft_position']['pieces']
    assert classifier.last_response['piece_reviews']


def test_light_empty_board_grid_is_not_ninety_pieces(classifier):
    image = np.full((1001, 901, 3), 225, np.uint8)
    for file in range(9):
        cv2.line(image, (50 + file * 100, 50), (50 + file * 100, 950), (25, 25, 25), 4)
    for rank in range(10):
        cv2.line(image, (50, 50 + rank * 100), (850, 50 + rank * 100), (25, 25, 25), 4)
    classifier.recognize_position([image], actual=True)
    assert not classifier.last_response['draft_position']['pieces']


@pytest.mark.parametrize('surface', [(225, 225, 225), (105, 170, 220)])
def test_round_object_without_printed_glyph_is_not_a_chess_kind(classifier, surface):
    image = physical_board([])
    cv2.circle(image, (250, 450), 41, (30, 30, 30), 3)
    cv2.circle(image, (250, 450), 38, surface, -1)
    assert classifier.recognize_position([image], actual=True) == []
    assert classifier.last_response['piece_reviews']
    assert not classifier.last_response['draft_position']['pieces']


@pytest.mark.parametrize('amplitude', [20, 40])
def test_smooth_shadow_on_unprinted_disc_is_not_a_glyph(classifier, amplitude):
    image = physical_board([])
    cv2.circle(image, (250, 450), 41, (30, 30, 30), 3)
    yy, xx = np.mgrid[:1001, :901]
    disk = (xx - 250) ** 2 + (yy - 450) ** 2 < 38 ** 2
    shade = (210 + (xx[disk] - 250) * amplitude / 38).astype(np.uint8)
    image[disk] = shade[:, None]
    assert classifier.recognize_position([image], actual=True) == []
    assert not classifier.last_response['draft_position']['pieces']


def test_high_probability_crop_disagreement_is_not_accepted(classifier, monkeypatch):
    calls = []
    def unstable(outputs, inputs):
        logits = np.zeros((4, 7), np.float32)
        logits[:, 1 if not calls else 3] = 10
        calls.append(True)
        return [logits]
    monkeypatch.setattr(classifier.session, 'run', unstable)
    glyph = np.full((96, 96), 220, np.uint8)
    cv2.putText(glyph, 'P', (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 20, 4)
    result = classifier.classify(glyph, 'black')
    assert result['prediction'] == 'rook'
    assert result['kind'] is None


def test_neighboring_frames_with_a_move_are_not_combined(classifier):
    images = [physical_board([(2, 4, 'red')], offset=0), physical_board([(2, 5, 'red')], offset=0)]
    classifier.recognize_position(images, actual=True)
    pieces = classifier.last_response['draft_position']['pieces']
    assert {(p['square']['file'], p['square']['rank']) for p in pieces} == {(2, 5)}
    assert classifier.last_response['piece_reviews']


def test_hidden_move_does_not_duplicate_one_piece_in_initial_draft(classifier):
    images = [physical_board([(2, 4, 'red'), (7, 5, 'black')], offset=0),
              physical_board([(2, 5, 'red'), (7, 5, 'black')], offset=0)]
    cv2.rectangle(images[0], (200, 500), (300, 600), (90, 130, 190), -1)
    cv2.rectangle(images[1], (200, 400), (300, 500), (90, 130, 190), -1)
    classifier.recognize_position(images + [images[-1]], actual=True)
    pieces = classifier.last_response['draft_position']['pieces']
    assert {(p['square']['file'], p['square']['rank']) for p in pieces} == {(2, 5), (7, 5)}
    assert any(r['square'] == {'file': 2, 'rank': 4} and r['state'] == 'occupancy'
               for r in classifier.last_response['piece_reviews'])


def test_standard_rules_mode_finishes_without_undefined_piece_list(classifier):
    assert classifier.recognize_position([physical_board([])], actual=False) == []


@pytest.mark.parametrize('transform,square', [(lambda x: x, (2, 4)),
    (lambda x: cv2.rotate(x, cv2.ROTATE_180), (6, 5)), (lambda x: cv2.flip(x, 1), (6, 4))])
def test_displaced_face_detection_is_symmetric_under_board_orientation(classifier, transform, square):
    image = transform(physical_board([(2, 4, 'red')], offset=-40))
    classifier.recognize_position([image], actual=True)
    assert len(classifier.last_response['draft_position']['pieces']) == 1
    piece = classifier.last_response['draft_position']['pieces'][0]
    assert (piece['square']['file'], piece['square']['rank']) == square


def test_shape_recentering_does_not_replace_a_clear_existing_glyph():
    from pathlib import Path
    from app.piece_classifier import PieceKindClassifier
    root = Path(__file__).resolve().parents[2]
    path = root / 'output/general-initial-recognition/case-7-endgame-5s.jpg'
    if not path.exists():
        pytest.skip('real endgame audit image unavailable')
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    model = PieceKindClassifier()
    model.recognize_position([image], actual=True)
    pieces = {(p['square']['file'], p['square']['rank']): p for p in model.last_response['draft_position']['pieces']}
    assert pieces[4, 0]['kind'] == 'pawn'
    assert pieces[0, 7]['kind'] == 'cannon'


def test_secondary_model_disagreement_marks_a_confident_glyph(classifier):
    class Layout:
        def readings(self, image):
            return {(2, 4): {'color': 'red', 'prediction': 'cannon', 'score': .95, 'margin': .8}}
    classifier.layout_classifier = Layout()
    classifier.recognize_position([physical_board([(2, 4, 'red')], offset=0)], actual=True)
    assert classifier.last_response['piece_reviews']
    assert any('模型' in r['reason'] for r in classifier.last_response['piece_reviews'])
    assert not classifier.last_response['draft_position']['pieces']
    assert classifier.last_response['piece_reviews'][0]['prediction'] is None


def test_secondary_model_cannot_invent_a_piece_on_empty_square(classifier):
    class Layout:
        def readings(self, image):
            return {(4, 4): {'color': 'black', 'prediction': 'king', 'score': .99, 'margin': .98}}
    classifier.layout_classifier = Layout()
    classifier.recognize_position([physical_board([])], actual=True)
    assert not classifier.last_response['draft_position']['pieces']


def test_initial_context_does_not_clip_top_row_glyph(tmp_path, monkeypatch):
    import app.initial as initial
    from app.schemas import Calibration, VideoMetadata
    from app.storage import TaskStore
    store = TaskStore(tmp_path)
    task = store.create('context.mp4', VideoMetadata(width=901, height=1001, duration_s=1, codec='h264'))
    task.calibration = Calibration(corners=[{'x': 50, 'y': 100}, {'x': 850, 'y': 100},
                                           {'x': 850, 'y': 900}, {'x': 50, 'y': 900}])
    task.calibration_confirmed = True
    task = store.save(task, task.revision)
    (store.task_dir(task.id) / 'evidence').mkdir()
    frame = physical_board([(4, 0, 'black')], offset=0)
    monkeypatch.setattr(initial, 'iter_frames', lambda *args: ((t, image) for t, image in [(0, frame)]))
    monkeypatch.setattr(initial, 'initial_frame_images', lambda *args: [args[-1]])
    class Vision:
        usage = []
        last_response = None
        def recognize_position(self, images, actual=False):
            assert images[0].shape == (1101, 1001, 3)
            return []
        def close(self): pass
    result = initial.run_initial_operation(task, store, Vision())
    assert result.stage != '起始画面识别未完成'
