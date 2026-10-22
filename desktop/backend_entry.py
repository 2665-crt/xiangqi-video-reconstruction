import argparse
import json
import multiprocessing
import os
from pathlib import Path
import socket
import sys


if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--recording-root', type=Path, required=True)
    parser.add_argument('--frontend-root', type=Path, required=True)
    parser.add_argument('--ready-file', type=Path, required=True)
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    logs = args.data_root / 'logs'
    logs.mkdir(exist_ok=True)
    sys.stdout = open(logs / 'backend.log', 'a', encoding='utf-8', buffering=1)
    sys.stderr = sys.stdout
    os.environ['XIANGQI_DATA_ROOT'] = str(args.data_root)
    os.environ['XIANGQI_RECORDING_ROOT'] = str(args.recording_root)
    os.environ['VISION_FUSION'] = '1'
    os.environ['MPLCONFIGDIR'] = str(args.data_root / 'matplotlib')
    os.environ['NUMBA_CACHE_DIR'] = str(args.data_root / 'numba')
    import uvicorn
    from app.main import app
    from app.desktop import configure_desktop_app
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    origin = f'http://127.0.0.1:{port}'
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='info', access_log=False))
    configure_desktop_app(app, args.frontend_root, origin, os.environ['XIANGQI_DESKTOP_TOKEN'], lambda: setattr(server, 'should_exit', True))
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.ready_file.with_suffix('.tmp')
    temporary.write_text(json.dumps({'url': origin, 'pid': os.getpid()}), encoding='utf-8')
    temporary.replace(args.ready_file)
    try:
        server.run(sockets=[listener])
    finally:
        listener.close()
        args.ready_file.unlink(missing_ok=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
