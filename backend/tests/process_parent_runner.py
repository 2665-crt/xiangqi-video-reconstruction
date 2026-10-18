import os
import time

from app.preprocessing import process_frames


def worker_pid(_):
    return os.getpid()


if __name__ == '__main__':
    with process_frames(iter([0]), worker_pid, workers=1, capacity=1) as frames:
        print(next(frames), flush=True)
        time.sleep(60)
