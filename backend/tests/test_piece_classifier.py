import hashlib
import json
import numpy as np
import pytest

@pytest.fixture
def classifier(tmp_path, monkeypatch):
    import onnxruntime as ort
    from app.piece_classifier import KINDS, PieceKindClassifier
    path = tmp_path / 'classifier.onnx'
    path.write_bytes(b'test-model')
    metadata = {'classes': list(KINDS), 'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'minimum_probability': .9, 'minimum_margin': .2, 'automatic_acceptance_enabled': True}
    path.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf8')
    class Session:
        def run(self, outputs, inputs):
            assert inputs['glyph'].shape == (4, 1, 48, 48)
            return [np.tile(np.array([[0, 10, 0, 0, 0, 0, 0]], np.float32), (4, 1))]
    monkeypatch.setattr(ort, 'InferenceSession', lambda *args, **kwargs: Session())
    return PieceKindClassifier(path)

def test_padding_and_outline_are_not_read_as_glyph_ink():
    import cv2
    from app.piece_classifier import glyph_image
    face = np.full((80, 80), 255, np.uint8)
    cv2.circle(face, (40, 40), 34, 180, -1)
    cv2.circle(face, (40, 40), 31, 35, 2)
    cv2.line(face, (40, 27), (40, 53), 25, 4)
    image = glyph_image(face)
    assert image.shape == (48, 48) and image.dtype == np.float32
    assert image.mean() < .35 and image[24, 24] > .8


def test_faint_real_piece_strokes_are_preserved_without_hard_threshold():
    import cv2
    from app.piece_classifier import glyph_image
    face = np.full((80, 80), 255, np.uint8)
    cv2.circle(face, (40, 40), 34, 180, -1)
    cv2.circle(face, (40, 40), 31, 35, 2)
    cv2.line(face, (40, 30), (40, 50), 173, 4)
    image = glyph_image(face)
    assert image[24, 24] > .1
    assert image[24, 18] < .02


def test_piece_crop_preserves_strokes_close_to_physical_face_edge():
    import cv2
    from app.piece_classifier import glyph_patch
    board = np.full((1001, 901, 3), (90, 150, 210), np.uint8)
    cv2.circle(board, (450, 350), 44, (235, 235, 235), -1)
    cv2.line(board, (412, 340), (412, 360), (20, 20, 20), 3)
    expected = np.count_nonzero(cv2.cvtColor(board, cv2.COLOR_BGR2GRAY) < 70)
    patch = glyph_patch(board, 4, 3, 'black')
    # Board rectification and subpixel centering soften dark strokes twice.
    assert np.count_nonzero(patch < 150) >= expected*.95


def test_letter_colour_ignores_red_print_outside_black_piece_face():
    import cv2
    from app.piece_classifier import piece_color
    image = np.full((96, 96, 3), 225, np.uint8)
    cv2.rectangle(image, (0, 0), (95, 7), (0, 0, 190), -1)
    cv2.line(image, (48, 28), (48, 68), (25, 25, 25), 5)
    assert piece_color(image) == 'black'
    cv2.line(image, (48, 28), (48, 68), (55, 55, 180), 5)
    assert piece_color(image) == 'red'


def test_piece_crop_does_not_jump_when_shadow_brightness_changes_slightly():
    import cv2
    from app.piece_classifier import glyph_image, glyph_patch
    images = []
    for shade in (149, 151):
        board = np.full((1001, 901, 3), (90, 150, 210), np.uint8)
        yy, xx = np.mgrid[:1001, :901]
        face = (xx-450)**2+(yy-350)**2 < 40**2
        board[face] = (205, 205, 205)
        board[face & (xx < 450)] = (shade, shade, shade)
        cv2.line(board, (450, 327), (450, 373), (25, 25, 25), 3)
        cv2.line(board, (430, 350), (470, 350), (25, 25, 25), 3)
        images.append(glyph_image(glyph_patch(board, 4, 3, 'black')))
    assert np.corrcoef(images[0].ravel(), images[1].ravel())[0, 1] > .95

def test_piece_crop_keeps_top_strokes_above_the_grid_tile():
    import cv2
    from app.piece_classifier import glyph_patch
    board = np.full((1000, 900, 3), (90, 150, 210), np.uint8)
    cv2.circle(board, (450, 325), 43, (235, 235, 235), -1)
    cv2.rectangle(board, (435, 294), (465, 299), (20, 20, 20), -1)
    cv2.line(board, (450, 299), (450, 340), (20, 20, 20), 3)
    patch = glyph_patch(board, 4, 3, 'black')
    assert np.count_nonzero(patch < 50, axis=1).max() >= 25


def test_cnn_keeps_color_separate_and_rejects_blank(classifier):
    glyph = np.full((30, 30), 255, np.uint8)
    glyph[7:23, 12:17] = 0
    assert classifier.classify(glyph, 'red')['kind'] == 'rook'
    assert classifier.classify(glyph, 'black')['kind'] == 'rook'
    assert classifier.classify(np.full((30, 30), 255, np.uint8), 'red')['kind'] is None

def test_unvalidated_model_cannot_accept_even_confident_output(classifier):
    classifier.metadata['automatic_acceptance_enabled'] = False
    glyph = np.full((30, 30), 255, np.uint8)
    glyph[7:23, 12:17] = 0
    result = classifier.classify(glyph, 'red')
    assert result['kind'] is None
    assert result['prediction'] == 'rook' and result['score'] > .99

def test_ambiguous_logits_do_not_become_piece_kind(classifier, monkeypatch):
    monkeypatch.setattr(classifier.session, 'run', lambda *args: [np.zeros((4, 7), np.float32)])
    glyph = np.full((30, 30), 255, np.uint8)
    glyph[7:23, 12:17] = 0
    assert classifier.classify(glyph, 'red')['kind'] is None


def test_four_rotations_reduce_one_confidently_wrong_orientation(classifier, monkeypatch):
    logits = np.zeros((4, 7), np.float32)
    logits[0, 6] = 10
    logits[1:, 1] = 10
    monkeypatch.setattr(classifier.session, 'run', lambda *args: [logits])
    glyph = np.full((30, 30), 255, np.uint8)
    glyph[7:23, 12:17] = 0
    result = classifier.classify(glyph, 'black')
    assert result['prediction'] == 'rook'
    assert .74 < result['score'] < .76
    assert result['kind'] is None


def test_known_color_selects_its_kind_head_without_switching_sides(classifier, monkeypatch):
    classifier.metadata['output_colors'] = ['red', 'black']
    logits = np.zeros((4, 14), np.float32)
    logits[:, 1] = 10
    logits[:, 13] = 10
    monkeypatch.setattr(classifier.session, 'run', lambda *args: [logits])
    glyph = np.full((30, 30), 255, np.uint8)
    glyph[7:23, 12:17] = 0
    assert classifier.classify(glyph, 'red')['kind'] == 'rook'
    assert classifier.classify(glyph, 'black')['kind'] == 'pawn'

def test_model_integrity_is_verified(tmp_path):
    from app.piece_classifier import KINDS, PieceKindClassifier
    path = tmp_path / 'model.onnx'
    path.write_bytes(b'changed')
    path.with_suffix('.json').write_text(json.dumps({'classes': list(KINDS), 'model_sha256': 'different'}))
    with pytest.raises(ValueError, match='校验'):
        PieceKindClassifier(path)

def test_local_initial_rejects_unknown_and_spends_no_cloud_request(classifier, monkeypatch):
    from app.schemas import opening_position
    from test_tracking import board_image
    monkeypatch.setattr(classifier, 'classify', lambda image, color: {'kind': None, 'score': .4, 'margin': .01})
    assert classifier.recognize_position([board_image(opening_position())], actual=True) == []
    assert classifier.last_response['uncertain_squares']
    assert '置信度不足' in classifier.last_response['error']
    assert classifier.last_request_count == 0 and classifier.last_image_count == 0

def test_disagreeing_frames_do_not_auto_confirm(classifier, monkeypatch):
    from app.schemas import Position, Piece, Square
    from test_tracking import board_image
    position = Position(pieces=[Piece(id='r', color='red', kind='rook', square=Square(file=4, rank=3))], side_to_move='red')
    readings = iter(['rook', 'cannon'])
    monkeypatch.setattr(classifier, 'classify', lambda image, color: {'kind': next(readings), 'score': .95, 'margin': .2})
    assert classifier.recognize_position([board_image(position), board_image(position)], actual=True) == []
    assert '连续画面' in classifier.last_response['error']

def test_initial_recognition_uses_local_classifier_without_http(classifier, monkeypatch):
    from app.vision import VisionClient
    from app.schemas import opening_position
    from test_tracking import board_image
    monkeypatch.setattr('app.piece_classifier.local_piece_classifier', lambda: classifier)
    for name in ('VISION_API_BASE', 'VISION_API_KEY', 'VISION_MODEL'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(classifier, 'recognize_position', lambda *args, **kwargs: [])
    vision = VisionClient()
    assert vision.configured
    assert vision.recognize_position([board_image(opening_position())], actual=True) == []
    assert vision.last_request_count == vision.last_image_count == 0
    vision.close()


def test_low_confidence_preserves_complete_editable_draft_without_auto_acceptance(classifier, monkeypatch):
    from app.schemas import Position, Piece, Square
    from test_tracking import board_image
    position = Position(pieces=[Piece(id='r', color='red', kind='rook', square=Square(file=4, rank=3)),
                                Piece(id='b', color='black', kind='cannon', square=Square(file=0, rank=7))], side_to_move='red')
    monkeypatch.setattr(classifier, 'classify', lambda image, color: {
        'kind': 'rook' if color == 'red' else None,
        'prediction': 'rook' if color == 'red' else 'cannon',
        'score': .98 if color == 'red' else .65, 'margin': .5})
    assert classifier.recognize_position([board_image(position)], actual=True) == []
    draft = Position.model_validate(classifier.last_response['draft_position'])
    assert {(p.square.file, p.square.rank, p.color, p.kind) for p in draft.pieces} == {
        (4, 3, 'red', 'rook'), (0, 7, 'black', 'cannon')}
    reviews = classifier.last_response['piece_reviews']
    assert len(reviews) == 1 and reviews[0]['square'] == {'file': 0, 'rank': 7}
    assert reviews[0]['prediction'] == 'cannon' and reviews[0]['score'] == .65
    assert classifier.last_request_count == 0


def test_disagreement_uses_best_reading_for_draft_but_marks_square_for_review(classifier, monkeypatch):
    from app.schemas import Position, Piece, Square
    from test_tracking import board_image
    position = Position(pieces=[Piece(id='r', color='red', kind='rook', square=Square(file=4, rank=3))], side_to_move='red')
    readings = iter([{'kind': None, 'prediction': 'cannon', 'score': .55, 'margin': .1},
                     {'kind': 'rook', 'prediction': 'rook', 'score': .97, 'margin': .7}])
    monkeypatch.setattr(classifier, 'classify', lambda *args: next(readings))
    assert classifier.recognize_position([board_image(position), board_image(position)], actual=True) == []
    assert classifier.last_response['draft_position']['pieces'][0]['kind'] == 'rook'
    assert classifier.last_response['piece_reviews'][0]['square'] == {'file': 4, 'rank': 3}


def test_blank_glyph_is_marked_without_inventing_a_piece_kind(classifier, monkeypatch):
    from app.schemas import Position, Piece, Square
    from test_tracking import board_image
    position = Position(pieces=[Piece(id='r', color='red', kind='rook', square=Square(file=4, rank=3))], side_to_move='red')
    monkeypatch.setattr(classifier, 'classify', lambda *args: {'kind': None, 'prediction': None, 'score': 0., 'margin': 0.})
    assert classifier.recognize_position([board_image(position)], actual=True) == []
    assert classifier.last_response['draft_position']['pieces'] == []
    assert classifier.last_response['piece_reviews'][0]['prediction'] is None
