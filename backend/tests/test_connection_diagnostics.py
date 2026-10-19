import httpx
import numpy as np
import pytest

from app.vision import VisionClient


def test_connection_trace_reports_tls_failure_without_credentials():
    def fail(request):
        trace = request.extensions['trace']
        trace('connection.connect_tcp.started', {})
        trace('connection.connect_tcp.complete', {})
        trace('connection.start_tls.started', {})
        trace('connection.start_tls.failed', {'exception': httpx.ConnectTimeout('tls timeout')})
        raise httpx.ConnectTimeout('tls timeout', request=request)
    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        vision = VisionClient('https://example.test/v1', 'do-not-log-this-key', 'test', client)
        with pytest.raises(ValueError):
            vision.recognize_position([np.zeros((100, 100, 3), np.uint8)])
        assert vision.last_response['connection_phase'] == 'tls'
        assert vision.last_response['trace'][-1]['event'] == 'connection.start_tls.failed'
        assert 'do-not-log-this-key' not in str(vision.last_response)
