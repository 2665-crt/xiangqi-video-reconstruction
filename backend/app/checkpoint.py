import json
import re
from pathlib import Path
from uuid import uuid4

import numpy as np


class CheckpointStore:
    def __init__(self, root):
        self.root = Path(root)

    def directory(self, task_id, generation):
        if not all(re.fullmatch('[a-f0-9]{32}', part) for part in (task_id, generation)):
            raise ValueError('恢复状态标识无效')
        return self.root / 'tasks' / task_id / 'recovery' / generation

    def save(self, task_id, generation, state, name=None, *, compress=True):
        directory = self.directory(task_id, generation)
        directory.mkdir(parents=True, exist_ok=True)
        name = name or uuid4().hex + '.json'
        if not re.fullmatch(r'[a-f0-9]{32,64}\.json', name):
            raise ValueError('恢复文件名无效')
        arrays = {}
        def encode(value):
            if isinstance(value, np.ndarray):
                key = 'array' + str(len(arrays))
                arrays[key] = value
                return {'__array__': key}
            if isinstance(value, dict):
                return {k: encode(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(v) for v in value]
            return value
        payload = {'version': 1, 'state': encode(state)}
        array_path = directory / (name[:-5] + '.npz')
        if arrays:
            with array_path.with_suffix('.tmp').open('wb') as stream:
                save_arrays = np.savez_compressed if compress else np.savez
                save_arrays(stream, **arrays)
            array_path.with_suffix('.tmp').replace(array_path)
        target = directory / name
        temporary = directory / (uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        temporary.replace(target)
        return name

    def load(self, task_id, generation, name):
        if not re.fullmatch(r'[a-f0-9]{32,64}\.json', name):
            raise ValueError('恢复文件名无效')
        directory = self.directory(task_id, generation)
        payload = json.loads((directory / name).read_text(encoding='utf-8'))
        if payload['version'] != 1:
            raise ValueError('恢复状态版本不兼容')
        path = directory / (name[:-5] + '.npz')
        arrays = {}
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
        def decode(value):
            if isinstance(value, dict) and set(value) == {'__array__'}:
                return arrays[value['__array__']]
            if isinstance(value, dict):
                return {k: decode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value
        return decode(payload['state'])
