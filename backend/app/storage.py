import json
import re
import sqlite3
from time import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from .schemas import Task, VideoMetadata


class RevisionConflict(ValueError):
    pass


class RecoveryCancelled(RevisionConflict):
    pass


class TaskStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "tasks.sqlite3"
        with closing(sqlite3.connect(self.db)) as con, con:
            con.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, payload TEXT NOT NULL)")

    def task_dir(self, task_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", task_id):
            raise KeyError(task_id)
        return self.root / "tasks" / task_id

    def create(self, filename: str, metadata: VideoMetadata) -> Task:
        task = Task(id=uuid4().hex, filename=filename, metadata=metadata, end_s=metadata.duration_s)
        self.task_dir(task.id).mkdir(parents=True)
        with closing(sqlite3.connect(self.db)) as con, con:
            con.execute("INSERT INTO tasks VALUES (?,?,?)", (task.id, task.revision, task.model_dump_json(by_alias=True)))
        return task

    def get(self, task_id: str) -> Task:
        with closing(sqlite3.connect(self.db)) as con:
            row = con.execute("SELECT payload FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return Task.model_validate_json(row[0])

    def list_tasks(self) -> list[Task]:
        with closing(sqlite3.connect(self.db)) as con:
            return [Task.model_validate_json(row[0]) for row in con.execute("SELECT payload FROM tasks ORDER BY rowid DESC")]

    def save(self, task: Task, expected_revision: int) -> Task:
        saved = task.model_copy(update={"revision": expected_revision + 1}, deep=True)
        with closing(sqlite3.connect(self.db)) as con, con:
            result = con.execute("UPDATE tasks SET revision=?,payload=? WHERE id=? AND revision=?",
                                 (saved.revision, saved.model_dump_json(by_alias=True), task.id, expected_revision))
            if result.rowcount != 1:
                raise RevisionConflict("任务已更新，请刷新后重试")
        return saved

    def save_analysis(self, task: Task) -> Task:
        with closing(sqlite3.connect(self.db)) as con, con:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute('SELECT payload FROM tasks WHERE id=?', (task.id,)).fetchone()
            if row is None:
                raise KeyError(task.id)
            latest = Task.model_validate_json(row[0])
            if task.recovery is not None:
                if (latest.recovery is None or not latest.recovery.auto_run
                        or latest.recovery.generation != task.recovery.generation
                        or latest.recovery.lease_token != task.recovery.lease_token):
                    raise RecoveryCancelled('当前分析已停止或任务版本已改变')
            saved = task.model_copy(deep=True)
            if task.recovery is not None:
                saved.recovery = latest.recovery.model_copy(deep=True)
                saved.recovery.heartbeat_at = time()
                saved.analysis_stats.model_requests = max(saved.analysis_stats.model_requests, saved.recovery.total_requests)
                saved.analysis_stats.images_sent = max(saved.analysis_stats.images_sent, saved.recovery.total_images)
                if saved.run_stats is not None:
                    saved.run_stats.model_requests = max(saved.run_stats.model_requests, saved.recovery.total_requests - saved.recovery.initial_requests)
                    saved.run_stats.images_sent = max(saved.run_stats.images_sent, saved.recovery.total_images - saved.recovery.initial_images)
                if saved.status in ('ready', 'needs_review', 'failed', 'action_required') or (saved.recovery.operation == 'initial' and saved.status not in ('analyzing', 'waiting_service')):
                    saved.recovery.auto_run = False
                    saved.recovery.lease_token = None
            for record in latest.moves:
                if record.status == 'corrected' and record.index <= len(saved.moves):
                    saved.moves[record.index - 1] = record.model_copy(deep=True)
            corrected = next((r.index for r in saved.moves if r.manual_override), None)
            if corrected is not None:
                from .turns import replay_corrections
                status, stage = saved.status, saved.stage
                saved = replay_corrections(saved, corrected)
                if status not in ('ready', 'needs_review'):
                    saved.status, saved.stage = status, stage
            saved.confirmed_prefix_length = next((r.index - 1 for r in saved.moves
                if r.status not in ('auto_verified', 'corrected')), len(saved.moves))
            if saved.status in ('ready', 'needs_review') and saved.initial_position is not None:
                saved.status = 'needs_review' if any(r.status not in ('auto_verified', 'corrected') for r in saved.moves) else 'ready'
                saved.stage = '校正已保存，规则问题仅提示' if corrected is not None else ('仅需审核标记步骤，后续清晰步骤已独立校验' if saved.status == 'needs_review' else '合法棋谱已保存，可回放')
            saved.revision = latest.revision + 1
            con.execute('UPDATE tasks SET revision=?,payload=? WHERE id=?',
                        (saved.revision, saved.model_dump_json(by_alias=True), saved.id))
        return saved

    def update_recovery(self, task_id, generation, lease_token, update):
        with closing(sqlite3.connect(self.db)) as con, con:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute('SELECT payload FROM tasks WHERE id=?', (task_id,)).fetchone()
            task = Task.model_validate_json(row[0])
            if (task.recovery is None or not task.recovery.auto_run or task.recovery.generation != generation
                    or task.recovery.lease_token != lease_token):
                raise RecoveryCancelled('当前分析已停止或任务版本已改变')
            update(task)
            task.revision += 1
            con.execute('UPDATE tasks SET revision=?,payload=? WHERE id=?',
                        (task.revision, task.model_dump_json(by_alias=True), task.id))
        return task

    def claim_recovery(self, task_id, generation, now):
        with closing(sqlite3.connect(self.db)) as con, con:
            con.execute('BEGIN IMMEDIATE')
            row = con.execute('SELECT payload FROM tasks WHERE id=?', (task_id,)).fetchone()
            task = Task.model_validate_json(row[0])
            r = task.recovery
            if (task.status != 'waiting_service' or r is None or not r.auto_run or r.generation != generation
                    or r.lease_token is not None or r.next_attempt_at > now):
                return None
            task.status, task.stage, task.error = 'analyzing', '后台正在恢复识别和分析', None
            r.lease_token = uuid4().hex
            r.heartbeat_at = now
            r.request_started_at = None
            task.revision += 1
            con.execute('UPDATE tasks SET revision=?,payload=? WHERE id=?',
                        (task.revision, task.model_dump_json(by_alias=True), task.id))
        return task

    def recover_interrupted(self):
        for task in self.list_tasks():
            if task.status == "analyzing":
                if task.recovery is not None and task.recovery.auto_run:
                    task.status, task.stage, task.error = 'waiting_service', '后台重启，正在自动恢复保存的任务', None
                    task.recovery.lease_token = None
                    task.recovery.next_attempt_at = max(time(), task.recovery.next_attempt_at)
                    self.save(task, task.revision)
                    continue
                if task.pending_turn is not None:
                    pending = task.pending_turn
                    pending.index, pending.status, pending.position_after = len(task.moves) + 1, 'needs_review', None
                    pending.reason = '上次分析中断，本回合尚未确认，请人工审核'
                    task.moves.append(pending)
                    task.pending_turn = None
                task.status, task.stage = "interrupted", "上次分析中断，可重新开始"
                self.save(task, task.revision)
