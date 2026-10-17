import base64
import json

import cv2
import httpx
import pytest
import numpy as np
from fastapi.testclient import TestClient


def test_missing_cloud_and_local_classifier_blocks_analysis(tmp_path, video_path, monkeypatch):
    from app.main import create_app
    monkeypatch.setattr('app.piece_classifier.local_piece_classifier', lambda: None)
    for name in ("VISION_API_BASE", "VISION_MODEL", "VISION_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with TestClient(create_app(tmp_path)) as client:
        with video_path.open("rb") as file:
            task = client.post("/api/tasks", files={"file": ("a.mp4", file, "video/mp4")}).json()
        result = client.post(f"/api/tasks/{task['id']}/analyze")
        assert result.status_code == 409
        assert "视觉模型" in result.json()["detail"]
        assert client.get(f"/api/tasks/{task['id']}").json()["moves"] == []


def test_vision_response_validation():
    from app.vision import parse_positions
    assert parse_positions('{"candidates":[{"pieces":[{"color":"red","kind":"dragon","file":9,"rank":0}],"side_to_move":"red"}]}') == []


def test_invalid_board_does_not_discard_valid_candidate():
    from app.vision import parse_positions
    invalid = {"pieces": [{"color": "red", "kind": "king", "file": 4, "rank": 6},
                          {"color": "black", "kind": "king", "file": 3, "rank": 0}], "side_to_move": None}
    valid = {"pieces": [{"color": "red", "kind": "king", "file": 4, "rank": 9},
                        {"color": "black", "kind": "king", "file": 3, "rank": 0}], "side_to_move": None}
    positions = parse_positions(json.dumps({"candidates": [invalid, valid]}))
    assert len(positions) == 2
    assert {p.side_to_move for p in positions} == {"red", "black"}
    assert all(p.pieces[0].square.rank == 9 for p in positions)


def test_vision_preserves_extra_cannons_in_demonstration_endgame():
    from app.vision import parse_positions
    pieces = [{"color": "red", "kind": "king", "file": 4, "rank": 9},
              {"color": "black", "kind": "king", "file": 4, "rank": 0}]
    pieces += [{"color": "black", "kind": "cannon", "file": file, "rank": rank}
               for file, rank in [(6, 0), (3, 2), (3, 5), (7, 9)]]
    positions = parse_positions(json.dumps({"candidates": [{"pieces": pieces, "side_to_move": None}]}))
    assert len(positions) == 2
    assert all(len(p.pieces) == 6 for p in positions)


@pytest.mark.parametrize("content", ['{"candidates":null}', '{"candidates":{}}', '{"pieces":[]}', 'null'])
def test_wrong_response_structure_has_explicit_error(content):
    from app.vision import parse_positions
    with pytest.raises(ValueError, match="candidates"):
        parse_positions(content)


def test_unusable_board_is_unknown_at_http_boundary():
    from app.vision import VisionClient
    payload = {"choices": [{"message": {"content": '{"candidates":[{"pieces":[],"side_to_move":null}]}'}}],
               "usage": {"total_tokens": 100}}
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as client:
        vision = VisionClient("https://example.test/v1", "placeholder", "test-vision", client=client)
        assert vision.recognize_position([np.zeros((100, 100, 3), np.uint8)]) == []
        assert vision.usage == [{"total_tokens": 100}]


def test_format_failure_preserves_usage_and_diagnostics():
    from app.vision import VisionClient
    payload = {"choices": [{"finish_reason": "length", "message": {"content": '{"candidates":['}}],
               "usage": {"total_tokens": 200}}
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as client:
        vision = VisionClient("https://example.test/v1", "placeholder", "test-vision", client=client)
        with pytest.raises(ValueError, match="JSON"):
            vision.recognize_position([np.zeros((100, 100, 3), np.uint8)])
        assert vision.usage == [{"total_tokens": 200}]
        assert vision.last_response["finish_reason"] == "length"
        assert vision.last_response["content"] == '{"candidates":['
        assert "placeholder" not in json.dumps(vision.last_response)


def test_initial_piece_sheet_prompt_ends_with_complete_piece_contract():
    from app.vision import VisionClient
    from test_tracking import board_image
    from app.schemas import opening_position

    def respond(request):
        body = json.loads(request.content)
        content = body['messages'][0]['content']
        assert len(content) == 3
        prompt = content[0]['text']
        assert '占位证据（仅供核对，不是输出格式）' in prompt
        assert '0,0: black' in prompt and '0,9: red' in prompt
        assert 'color、kind、file、rank' in prompt
        assert prompt.endswith('{"candidates":[{"pieces":[{"color":"red","kind":"king","file":4,"rank":9},{"color":"black","kind":"king","file":4,"rank":0}],"side_to_move":null}]}')
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"candidates":[]}'}}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        vision = VisionClient('https://example.test/v1', 'placeholder', 'test-vision', client)
        assert vision.recognize_position([board_image(opening_position())], actual=True) == []


def test_occupancy_only_response_explains_missing_kind_and_does_not_retry():
    from app.vision import VisionClient
    response = json.dumps([{'file': 4, 'rank': 0, 'color': 'black'},
                           {'file': 4, 'rank': 9, 'color': 'red'}])
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={'choices': [{'message': {'content': response}}],
                                        'usage': {'total_tokens': 42}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        vision = VisionClient('https://example.test/v1', 'placeholder', 'test-vision', client)
        with pytest.raises(ValueError, match='缺少棋种kind，无法区分车、炮、兵等棋子'):
            vision.recognize_position([np.zeros((100, 100, 3), np.uint8)], actual=True)
        assert len(calls) == 1
        assert vision.last_response['content'] == response
        assert vision.usage == [{'total_tokens': 42}]


def test_midgame_and_unknown_turn_preserved():
    from app.vision import parse_positions
    positions = parse_positions('{"candidates":[{"pieces":[{"color":"red","kind":"king","file":4,"rank":9},{"color":"black","kind":"king","file":3,"rank":0},{"color":"black","kind":"pawn","file":7,"rank":5}],"side_to_move":null}]}')
    assert {p.side_to_move for p in positions} == {"red", "black"}
    assert len(positions[0].pieces) == 3


def test_vision_actual_http_boundary():
    from app.vision import VisionClient
    content = '{"candidates":[{"pieces":[{"color":"red","kind":"king","file":4,"rank":9},{"color":"black","kind":"king","file":3,"rank":0}],"side_to_move":"red"}]}'
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": content}}], "usage": {"total_tokens": 80}})))
    vision = VisionClient("https://example.test/v1", "placeholder", "test-vision", client=client)
    positions = vision.recognize_position([np.zeros((100, 100, 3), np.uint8)])
    assert len(positions[0].pieces) == 2
    assert vision.usage == [{"total_tokens": 80}]


def test_qwen_omni_request_and_final_content():
    from app.vision import VisionClient

    def respond(request):
        assert str(request.url) == "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "qwen3.8-omni-flash"
        assert body["reasoning_effort"] == "none"
        assert body["response_format"] == {"type": "json_object"}
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert len(content) == 4
        for item, value in zip(content[1:], [40, 80, 120]):
            url = item["image_url"]["url"]
            assert url.startswith("data:image/jpeg;base64,")
            decoded = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
            assert decoded.shape == (100, 100, 3)
            assert abs(float(decoded.mean()) - value) < 2
        return httpx.Response(200, text='data: {"choices":[{"delta":{"reasoning_content":"不能作为棋局解析的思考文本"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"{\\\"candidates\\\":"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"[]}"},"finish_reason":"stop"}]}\n\n'
            'data: {"choices":[],"usage":{"total_tokens":120}}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        vision = VisionClient("https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/", "placeholder",
                              "qwen3.8-omni-flash", client=client)
        assert vision.recognize_position([np.full((100, 100, 3), value, np.uint8) for value in [0, 40, 80, 120]]) == []
        assert vision.usage == [{"total_tokens": 120}]
def test_connection_failure_has_safe_diagnostic_without_credentials():
    import httpx
    import numpy as np
    import pytest
    from app.vision import VisionClient
    def fail(request):
        raise httpx.ConnectError("socket forbidden", request=request)
    client = httpx.Client(transport=httpx.MockTransport(fail))
    vision = VisionClient("https://example.test/v1", "private-check-key", "test-vision", client=client)
    with pytest.raises(ValueError, match="连接失败"):
        vision.recognize_position([np.zeros((100, 100, 3), np.uint8)])
    assert vision.last_response["request_error"] == "ConnectError"
    assert "private-check-key" not in str(vision.last_response)
    vision.close()


def test_complete_flat_piece_response_is_accepted_without_inventing_pieces():
    from app.vision import parse_positions
    pieces = [{"color": "red", "kind": "king", "file": 4, "rank": 9},
              {"color": "black", "kind": "king", "file": 3, "rank": 0},
              {"color": "red", "kind": "pawn", "file": 4, "rank": 5}]
    positions = parse_positions(json.dumps(pieces))
    assert len(positions) == 2
    assert {p.side_to_move for p in positions} == {"red", "black"}
    assert all(len(p.pieces) == 3 for p in positions)
    with pytest.raises(ValueError):
        parse_positions('[1,2,3]')
    assert parse_positions(json.dumps(pieces[:1])) == []


@pytest.mark.parametrize("moved", [False, True])
def test_confirmed_reference_recovers_king_letter_errors_without_guessing_coordinates(moved):
    from app.chess import apply_move, legal_moves
    from app.schemas import opening_position
    from app.vision import VisionClient
    initial = opening_position()
    move = next(m for m in legal_moves(initial) if m.source.file == 4 and m.source.rank == 6)
    expected = apply_move(initial, move) if moved else initial
    pieces = [{"color": p.color, "kind": "rook" if p.kind == "king" else p.kind,
               "file": p.square.file, "rank": p.square.rank} for p in expected.pieces]
    content = json.dumps(pieces)
    payload = {"choices": [{"message": {"content": content}}]}
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as client:
        vision = VisionClient("https://example.test/v1", "placeholder", "test-vision", client)
        assert vision.recognize_position([np.zeros((100, 100, 3), np.uint8)], reference=initial) == [expected]


def test_reference_does_not_fill_missing_or_duplicate_model_coordinates():
    from app.schemas import opening_position
    from app.vision import parse_positions
    initial = opening_position()
    pieces = [{"color": p.color, "kind": "rook" if p.kind == "king" else p.kind,
               "file": p.square.file, "rank": p.square.rank} for p in initial.pieces]
    assert parse_positions(json.dumps(pieces[:-1]), reference=initial) == []
    assert parse_positions(json.dumps(pieces + [pieces[-1]]), reference=initial) == []


@pytest.mark.parametrize('exception, message', [
    (httpx.ConnectTimeout, '建立连接超时'),
    (httpx.ReadTimeout, '等待模型响应超时'),
    (httpx.WriteTimeout, '发送棋盘截图超时'),
])
def test_timeout_phase_is_explicit_and_request_is_not_retried(exception, message):
    from app.vision import VisionClient
    calls = []
    def fail(request):
        calls.append(request)
        raise exception('private-check-key', request=request)
    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        vision = VisionClient('https://example.test/v1', 'private-check-key', 'qwen3.8-omni-flash', client)
        with pytest.raises(ValueError, match=message):
            vision.recognize_position([np.zeros((100, 100, 3), np.uint8)])
        assert len(calls) == 1
        assert vision.last_response['request_error'] == exception.__name__
        assert 'elapsed_s' in vision.last_response
        assert 'private-check-key' not in json.dumps(vision.last_response)


def test_stream_failure_preserves_partial_usage_without_accepting_incomplete_position():
    from app.vision import VisionClient
    text = 'data: {"choices":[],"usage":{"total_tokens":15}}\n\n' + \
        'data: {"choices":[{"delta":{"content":"{\\\"candidates\\\":[]}"}}]}\n\n'
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=text))) as client:
        vision = VisionClient('https://example.test/v1', 'placeholder', 'qwen3.8-omni-flash', client)
        with pytest.raises(ValueError, match='流式响应中断'):
            vision.recognize_position([np.zeros((100, 100, 3), np.uint8)])
        assert vision.usage == [{'total_tokens': 15}]


def test_model_read_timeout_is_separate_from_connection_timeout():
    from app.vision import VisionClient
    vision = VisionClient('https://example.test/v1', 'placeholder', 'qwen3.8-omni-flash')
    try:
        assert vision.client.timeout.connect == 15
        assert vision.client.timeout.read == 180
    finally:
        vision.close()


def test_stream_can_complete_after_old_90_second_limit(monkeypatch):
    import app.vision as module
    elapsed = [0]
    monkeypatch.setattr(module, 'monotonic', lambda: elapsed[0])
    class SlowStream(httpx.SyncByteStream):
        def __iter__(self):
            elapsed[0] = 20
            yield b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'
            elapsed[0] = 100
            yield b'data: {"choices":[{"delta":{"content":"{\\\"candidates\\\":[]}"},"finish_reason":"stop"}]}\n\n'
            yield b'data: {"choices":[],"usage":{"total_tokens":25}}\n\ndata: [DONE]\n\n'
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=SlowStream()))) as client:
        vision = module.VisionClient('https://example.test/v1', 'placeholder', 'qwen3.8-omni-flash', client)
        assert vision.recognize_position([np.zeros((100, 100, 3), np.uint8)]) == []
        assert vision.usage == [{'total_tokens': 25}]
        assert vision.last_response['content'] == '{"candidates":[]}'


def test_stream_total_wait_is_bounded_and_does_not_resubmit(monkeypatch):
    import app.vision as module
    elapsed, calls, closed = [0], [], []
    monkeypatch.setattr(module, 'monotonic', lambda: elapsed[0])
    class EndlessStream(httpx.SyncByteStream):
        def __iter__(self):
            for i in range(1, 10):
                elapsed[0] = 50 * i
                yield b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'
        def close(self):
            closed.append(True)
    def respond(req):
        calls.append(req)
        return httpx.Response(200, stream=EndlessStream())
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        vision = module.VisionClient('https://example.test/v1', 'placeholder', 'qwen3.8-omni-flash', client)
        with pytest.raises(ValueError, match='等待模型响应超时'):
            vision.recognize_position([np.zeros((100, 100, 3), np.uint8)])
        assert len(calls) == len(closed) == 1
        assert vision.last_response['request_error'] == 'ReadTimeout'


