import json

import httpx
import numpy as np
import pytest

from app.vision import VisionClient


def test_remote_disconnect_preserves_phase_partial_usage_and_safe_diagnostic():
    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"choices":[],"usage":{"total_tokens":15}}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"{\\\"candidates\\\":"}}]}\n\n'
            raise httpx.RemoteProtocolError('secret-for-test')
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=BrokenStream(), headers={'x-request-id': 'req-123'}))) as client:
        vision = VisionClient('https://example.test/v1', 'secret-for-test', 'qwen3.8-omni-flash', client)
        with pytest.raises(ValueError) as caught:
            vision.recognize_position([np.zeros((100, 100, 3), np.uint8)])
        error = caught.value
        assert getattr(error, 'phase', None) == 'response'
        assert error.retryable and error.outcome_uncertain
        assert vision.last_response['received_chars'] > 0
        assert vision.last_response['request_id'] == 'req-123'
        assert vision.usage == [{'total_tokens': 15}]
        assert 'secret-for-test' not in json.dumps(vision.last_response) + str(error)


def test_connect_timeout_is_retryable_without_uncertain_inference():
    def fail(request):
        raise httpx.ConnectTimeout('secret-for-test', request=request)
    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        vision = VisionClient('https://example.test/v1', 'secret-for-test', 'qwen3.8-omni-flash', client)
        with pytest.raises(ValueError) as caught:
            vision.recognize_position([np.zeros((100, 100, 3), np.uint8)])
        assert getattr(caught.value, 'phase', None) == 'connect'
        assert caught.value.retryable and not caught.value.outcome_uncertain
        assert vision.last_response['response_started'] is False


@pytest.mark.parametrize('status,retryable', [(401, False), (403, False), (429, True), (503, True)])
def test_http_status_selects_recovery_and_retry_after(status, retryable):
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status, headers={'retry-after': '90'}))) as client:
        vision = VisionClient('https://example.test/v1', 'secret-for-test', 'test-model', client)
        with pytest.raises(ValueError) as caught:
            vision.recognize_position([np.zeros((100, 100, 3), np.uint8)])
        assert getattr(caught.value, 'retryable', None) is retryable
        assert caught.value.retry_after_s == 90


def test_incomplete_stream_is_retryable_but_not_accepted_as_position():
    text = 'data: {"choices":[{"delta":{"content":"{\\\"candidates\\\":[]}"}}]}\n\n'
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=text))) as client:
        vision = VisionClient('https://example.test/v1', 'secret-for-test', 'qwen3.8-omni-flash', client)
        with pytest.raises(ValueError) as caught:
            vision.recognize_position([np.zeros((100, 100, 3), np.uint8)])
        assert getattr(caught.value, 'code', None) == 'IncompleteStream'
        assert caught.value.outcome_uncertain


def test_gateway_timeout_consumes_uncertain_inference_budget():
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(504))) as client:
        vision = VisionClient('https://example.test/v1', 'secret-for-test', 'test-model', client)
        with pytest.raises(ValueError) as caught:
            vision.recognize_position([np.zeros((100, 100, 3), np.uint8)])
        assert caught.value.outcome_uncertain


def test_truncated_sse_data_line_enters_automatic_recovery_with_phase():
    text = 'data: {"choices":[],"usage":{"total_tokens":7}}\n\ndata: {"choices":['
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=text))) as client:
        vision = VisionClient('https://example.test/v1', 'secret-for-test', 'qwen3.8-omni-flash', client)
        with pytest.raises(ValueError) as caught:
            vision.recognize_position([np.zeros((100, 100, 3), np.uint8)])
        assert getattr(caught.value, 'outcome_uncertain', False)
        assert vision.last_response['response_started']
        assert vision.usage == [{'total_tokens': 7}]
