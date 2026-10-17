export type Color = 'red' | 'black';
export type PieceKind = 'king' | 'advisor' | 'elephant' | 'horse' | 'rook' | 'cannon' | 'pawn';
export interface Square { file: number; rank: number }
export interface Piece { id: string; color: Color; kind: PieceKind; square: Square }
export interface Position { pieces: Piece[]; side_to_move: Color }
export interface InitialPieceReview { square: Square; color: Color | null; prediction: PieceKind | null; score: number; reason: string; state?: 'kind' | 'occupancy' }
export interface Move { index: number; color: Color; piece_id: string; from: Square; to: Square; captured_piece_id: string | null }
export interface MoveRecord {
  index: number; move: Move | null; video_start_s: number; video_end_s: number;
  status: 'auto_verified' | 'needs_review' | 'corrected' | 'tentative';
  position_after: Position | null; evidence_ids: string[]; candidates: Move[]; reason: string;
  position_before?: Position | null;
  change_type?: 'move' | 'setup' | 'review'; observed_positions?: Position[];
  violation?: 'illegal_move' | 'multiple_pieces' | 'takeback' | 'turn_mismatch' | null;
  violation_note?: string; related_steps?: number[];
  manual_override?: boolean;
}
export interface Calibration { corners: { x: number; y: number }[]; orientation: 'normal' | 'rotated_180' | 'mirrored' | 'mirrored_rotated_180' }
export interface Task {
  id: string; filename: string; status: 'uploaded' | 'analyzing' | 'needs_review' | 'ready' | 'failed' | 'interrupted' | 'waiting_service' | 'action_required';
  stage: string; processed_video_s: number; start_s: number; end_s: number;
  metadata: { width: number; height: number; duration_s: number; codec: string };
  calibration: Calibration | null; initial_position: Position | null; initial_candidates: Position[];
  calibration_confirmed?: boolean;
  scan_progress_s?: number; scan_complete?: boolean; scan_events?: number;
  recognition_video_s?: number; recognition_events?: number; pending_event_count?: number;
  rebuild_video_s?: number | null; rebuild_target_s?: number | null;
  initial_draft?: Position | null; initial_piece_reviews?: InitialPieceReview[];
  initial_evidence_ids: string[]; moves: MoveRecord[]; confirmed_prefix_length: number; revision: number;
  initial_frame_s: number | null;
  analysis_stats: { scanned_frames: number; stable_events: number; uncertain_events: number; model_requests: number; images_sent: number };
  run_stats?: { scanned_frames: number; model_requests: number; images_sent: number; elapsed_seconds: number; initialization_seconds?: number; decode_seconds: number; opencv_seconds: number; model_seconds: number } | null;
  error: string | null; usage: Record<string, unknown>[];
  replay_mode?: 'legal' | 'actual';
  rules_enabled?: boolean;
  recovery?: { auto_run: boolean; operation: 'analyze' | 'initial'; attempts: number; uncertain_attempts: number;
    next_attempt_at: number; waiting_since: number | null; last_error: string | null; phase: string | null;
    total_requests: number; checkpoint: string | null; heartbeat_at?: number;
    request_started_at?: number | null; pending_request?: string | null } | null;
}
export interface TaskSummary { id: string; filename: string; status: Task['status']; steps: number }
export interface RecordingJob {
  id: string; task_id: string; revision: number; interval_s: number;
  status: 'queued' | 'running' | 'completed' | 'failed'; stage: string; progress: number;
  completed_frames: number; total_frames: number; duration_s: number; width: number; height: number; fps: number;
  filename: string; file_path: string | null; error: string | null; created_at: string;
}
export const pieceText: Record<Color, Record<PieceKind, string>> = {
  red: { king: '帅', advisor: '仕', elephant: '相', horse: '马', rook: '车', cannon: '炮', pawn: '兵' },
  black: { king: '将', advisor: '士', elephant: '象', horse: '马', rook: '车', cannon: '炮', pawn: '卒' },
};
export const timeText = (seconds: number) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
