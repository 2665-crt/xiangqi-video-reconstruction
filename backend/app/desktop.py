import hmac
import os
from pathlib import Path
import shutil

from fastapi import Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def recording_command(manifest_path, render_url):
    executable = os.getenv('XIANGQI_DESKTOP_EXE')
    if executable:
        return [executable, '--recording-render', str(manifest_path), render_url]
    node = shutil.which('node')
    if node is None:
        raise RuntimeError('未找到 Node.js，请检查本项目运行环境')
    root = Path(__file__).resolve().parents[2]
    return [node, str(root / 'scripts/render-recording.cjs'), str(manifest_path), render_url]


def configure_desktop_app(app, frontend_root, origin, token, shutdown):
    frontend_root = Path(frontend_root)
    app.state.recordings.render_url = origin + '/recording-render'

    @app.post('/api/desktop/shutdown', status_code=202)
    def request_shutdown(x_desktop_token: str = Header(default='')):
        if not hmac.compare_digest(x_desktop_token, token):
            raise HTTPException(403, '无效的桌面会话')
        shutdown()
        return {'status': 'stopping'}

    @app.get('/recording-render', include_in_schema=False)
    def recording_page():
        return FileResponse(frontend_root / 'index.html')

    app.mount('/', StaticFiles(directory=frontend_root, html=True), name='desktop-interface')
