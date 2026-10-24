import argparse
import json
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import threading


if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w', encoding='utf-8')


class DataDirectoryBusy(OSError):
    pass


def acquire_data_lock(data_root):
    import msvcrt
    runtime = data_root / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    handle = open(runtime / 'backend.lock', 'a+b')
    try:
        if handle.seek(0, 2) == 0:
            handle.write(b'\0')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise DataDirectoryBusy('实例数据目录正在使用') from exc
    except OSError:
        handle.close()
        raise
    return handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--recording-root', type=Path, required=True)
    parser.add_argument('--frontend-root', type=Path, required=True)
    parser.add_argument('--ready-file', type=Path, required=True)
    args = parser.parse_args()
    try:
        data_lock = acquire_data_lock(args.data_root)
    except DataDirectoryBusy:
        raise SystemExit(73)
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
    watch_stop = threading.Event()
    watcher = None
    if os.getenv('XIANGQI_DESKTOP_PID'):
        import psutil
        parent = psutil.Process(int(os.environ['XIANGQI_DESKTOP_PID']))

        def watch_desktop():
            while not watch_stop.wait(1):
                if not parent.is_running():
                    server.should_exit = True
                    break

        watcher = threading.Thread(target=watch_desktop, name='desktop-owner-watch', daemon=True)
        watcher.start()
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.ready_file.with_suffix('.tmp')
    temporary.write_text(json.dumps({'url': origin, 'pid': os.getpid(), 'instance_id': os.getenv('XIANGQI_INSTANCE_ID')}), encoding='utf-8')
    temporary.replace(args.ready_file)
    try:
        server.run(sockets=[listener])
    finally:
        watch_stop.set()
        if watcher:
            watcher.join(timeout=2)
        listener.close()
        args.ready_file.unlink(missing_ok=True)
        data_lock.close()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    if len(sys.argv) == 3 and sys.argv[1] == '--probe-data-root':
        try:
            with acquire_data_lock(Path(sys.argv[2])):
                pass
        except DataDirectoryBusy:
            raise SystemExit(73)
    else:
        main()
