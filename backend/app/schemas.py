from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Color = Literal["red", "black"]
PieceKind = Literal["king", "advisor", "elephant", "horse", "rook", "cannon", "pawn"]


class Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="forbid")


class Square(Model):
    file: int = Field(ge=0, le=8)
    rank: int = Field(ge=0, le=9)


class Piece(Model):
    id: str
    color: Color
    kind: PieceKind
    square: Square


class Position(Model):
    pieces: list[Piece]
    side_to_move: Color

    @model_validator(mode="after")
    def validate_board(self):
        squares = [(p.square.file, p.square.rank) for p in self.pieces]
        if len(set(squares)) != len(squares) or len({p.id for p in self.pieces}) != len(self.pieces):
            raise ValueError("棋子身份或位置重复")
        return self


class Move(Model):
    index: int = Field(ge=1)
    color: Color
    piece_id: str
    source: Square = Field(alias="from")
    target: Square = Field(alias="to")
    captured_piece_id: str | None = None


class Point(Model):
    x: float
    y: float


class Calibration(Model):
    corners: list[Point] = Field(min_length=4, max_length=4)
    orientation: Literal["normal", "rotated_180", "mirrored", "mirrored_rotated_180"] = "normal"


class VideoMetadata(Model):
    width: int
    height: int
    duration_s: float
    codec: str


class PieceEdit(Model):
    id: str
    removed: bool = False
    color: Color | None = None
    kind: PieceKind | None = None
    square: Square | None = None


class BoardCheck(Model):
    status: Literal['verified', 'partial', 'conflict']
    visible_before: int
    visible_after: int
    before_mismatches: list[Square] = Field(default_factory=list)
    after_mismatches: list[Square] = Field(default_factory=list)
    unknown_before: list[Square] = Field(default_factory=list)
    unknown_after: list[Square] = Field(default_factory=list)
    reason: str = ''
    backfilled: bool = False


class MoveRecord(Model):
    index: int = Field(ge=1)
    move: Move | None = None
    video_start_s: float
    video_end_s: float
    action_start_s: float | None = Field(default=None, ge=0)
    action_end_s: float | None = Field(default=None, ge=0)
    timing_status: Literal['legacy', 'observed', 'uncertain', 'estimated'] = 'legacy'
    status: Literal["auto_verified", "needs_review", "corrected", "tentative"]
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_times_s: list[float | None] = Field(default_factory=list)
    board_check: BoardCheck | None = None
    position_after: Position | None = None
    predicted_position: Position | None = None
    inference_score: float | None = None
    depends_on_steps: list[int] = Field(default_factory=list)
    position_before: Position | None = None
    tracking_anchor: Position | None = None
    tracking_anchor_s: float | None = Field(default=None, ge=0)
    observed_positions: list[Position] = Field(default_factory=list)
    review_changed_piece_ids: list[str] = Field(default_factory=list)
    review_witness_ids: list[str] = Field(default_factory=list)
    candidates: list[Move] = Field(default_factory=list)
    reason: str = ""
    inference_group: str | None = None
    change_type: Literal["move", "setup", "review"] = "move"
    violation: Literal["illegal_move", "multiple_pieces", "takeback", "turn_mismatch"] | None = None
    violation_note: str = ""
    related_steps: list[int] = Field(default_factory=list)
    position_edits: list[PieceEdit] | None = None
    position_side_to_move: Color | None = None
    manual_override: bool = False


class AnalysisStats(Model):
    scanned_frames: int = 0
    backfill_intervals: int | None = None
    backfill_frames: int = 0
    backfill_seconds: float = 0
    reanchor_checks: int = 0
    reanchors: int = 0
    reanchor_seconds: float = 0
    stable_events: int = 0
    uncertain_events: int = 0
    model_requests: int = 0
    images_sent: int = 0
    decode_seconds: float = 0
    initialization_seconds: float = 0
    opencv_seconds: float = 0
    model_seconds: float = 0
    elapsed_seconds: float = 0


class InitialPieceReview(Model):
    square: Square
    color: Color | None
    prediction: PieceKind | None
    score: float
    reason: str
    state: Literal['kind', 'occupancy'] = 'kind'


class RecoveryState(Model):
    auto_run: bool = True
    operation: Literal['analyze', 'initial'] = 'analyze'
    generation: str
    checkpoint: str | None = None
    pending_request: str | None = None
    attempts: int = 0
    uncertain_attempts: int = 0
    next_attempt_at: float = 0
    retry_not_before: float = 0
    waiting_since: float | None = None
    last_error: str | None = None
    phase: str | None = None
    lease_token: str | None = None
    local_only: bool = False
    total_requests: int = 0
    total_images: int = 0
    initial_requests: int = 0
    initial_images: int = 0
    initial_status: str = 'uploaded'
    origin_task_id: str | None = None
    origin_revision: int | None = None
    heartbeat_at: float = 0
    request_started_at: float | None = None


class RecheckState(Model):
    id: str
    status: Literal['running', 'ready', 'needs_review', 'applied', 'cancelled', 'failed'] = 'running'
    left_index: int
    right_index: int
    start_s: float
    end_s: float
    processed_s: float
    scanned_frames: int = 0
    records: list[MoveRecord] = Field(default_factory=list)
    endpoint_matches: bool | None = None
    unresolved: int = 0
    added: int = 0
    changed: int = 0
    removed: int = 0
    manual_conflicts: list[int] = Field(default_factory=list)
    integrated: bool = False
    integrated_count: int = 0
    message: str = '正在逐帧核对所选区间'


class Task(Model):
    id: str
    filename: str
    status: Literal["uploaded", "analyzing", "needs_review", "ready", "failed", "interrupted", "waiting_service", "action_required"] = "uploaded"
    stage: str = "视频已上传"
    processed_video_s: float = 0
    metadata: VideoMetadata
    start_s: float = 0
    end_s: float
    evidence_end_s: float | None = None
    calibration: Calibration | None = None
    calibration_confirmed: bool = False
    initial_position: Position | None = None
    context_position: Position | None = None
    initial_evidence_ids: list[str] = Field(default_factory=list)
    initial_candidates: list[Position] = Field(default_factory=list)
    initial_draft: Position | None = None
    initial_piece_reviews: list[InitialPieceReview] = Field(default_factory=list)
    initial_frame_s: float | None = None
    moves: list[MoveRecord] = Field(default_factory=list)
    confirmed_prefix_length: int = 0
    revision: int = 0
    error: str | None = None
    usage: list[dict] = Field(default_factory=list)
    analysis_stats: AnalysisStats = Field(default_factory=AnalysisStats)
    run_stats: AnalysisStats | None = None
    replay_mode: Literal["legal", "actual"] = "legal"
    rules_enabled: bool = False
    pending_turn: MoveRecord | None = None
    initial_turn_known: bool = True
    recovery: RecoveryState | None = None
    scan_progress_s: float = 0
    scan_complete: bool = False
    scan_events: int = 0
    recognition_video_s: float = 0
    recognition_events: int = 0
    pending_event_count: int = 0
    resume_video_s: float | None = None
    resume_calibration: Calibration | None = None
    resume_motion_active: bool = False
    resume_motion_reference_s: float | None = None
    resume_motion_reference_calibration: Calibration | None = None
    tracking_state: Literal['tracking', 'recovering'] = 'tracking'
    tracking_since_s: float | None = None
    tracking_message: str | None = None
    rebuild_video_s: float | None = None
    rebuild_target_s: float | None = None
    recheck: RecheckState | None = None


class InitialRecognitionRequest(Model):
    revision: int


class RecordingRequest(Model):
    revision: int
    interval_s: Literal[1, 2, 3] = 2


class SetupRequest(Model):
    revision: int
    start_s: float
    end_s: float
    calibration: Calibration | None = None
    initial_position: Position | None = None


class CorrectionRequest(Model):
    revision: int
    move: Move | None = None
    moves: list[Move] | None = Field(default=None, max_length=64)
    video_time_s: float | None = Field(default=None, ge=0)
    position: Position | None = None
    action_start_s: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def one_correction_format(self):
        if sum(value is not None for value in (self.move, self.moves, self.position)) != 1:
            raise ValueError("请选择单步、区间或完整局面校正")
        return self


class InsertRequest(Model):
    revision: int
    move: Move
    video_time_s: float = Field(ge=0)
    action_start_s: float | None = Field(default=None, ge=0)


class RecheckRequest(Model):
    revision: int
    first_index: int = Field(ge=1)
    second_index: int = Field(ge=1)


class RecheckActionRequest(Model):
    revision: int
    recheck_id: str


def opening_position() -> Position:
    pieces = []
    back = ["rook", "horse", "elephant", "advisor", "king", "advisor", "elephant", "horse", "rook"]
    for color, rank, cannon_rank, pawn_rank in [("black", 0, 2, 3), ("red", 9, 7, 6)]:
        for file, kind in enumerate(back):
            pieces.append(Piece(id=f"{color}_{kind}_{file}", color=color, kind=kind,
                                square=Square(file=file, rank=rank)))
        for file in (1, 7):
            pieces.append(Piece(id=f"{color}_cannon_{file}", color=color, kind="cannon",
                                square=Square(file=file, rank=cannon_rank)))
        for file in (0, 2, 4, 6, 8):
            pieces.append(Piece(id=f"{color}_pawn_{file}", color=color, kind="pawn",
                                square=Square(file=file, rank=pawn_rank)))
    return Position(pieces=pieces, side_to_move="red")
