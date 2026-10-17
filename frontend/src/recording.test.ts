import { describe, expect, it } from 'vitest';
import { recordingBlockReason } from './components/RecordingPanel';
import type { Task } from './types';

const task = { status: 'ready', start_s: 10, end_s: 120, processed_video_s: 120,
  initial_position: { side_to_move: 'red', pieces: [] },
  moves: [{ index: 1, status: 'auto_verified', position_after: { pieces: [] } }],
} as unknown as Task;
describe('录像导出入口', () => {
  it('完整确认的选区可以导出', () => expect(recordingBlockReason(task)).toBe(''));
  it('指出具体待复核步骤', () => expect(recordingBlockReason({ ...task, moves: [{ ...task.moves[0], index: 7, status: 'needs_review' }] })).toContain('第 7'));
  it('校正后仍未分析到选区末尾时不能导出', () => expect(recordingBlockReason({ ...task, processed_video_s: 70 })).toContain('完整'));
  it('未确认的起始候选不能作为录像开局', () => expect(recordingBlockReason({ ...task, initial_position: null })).toContain('初始'));
  it('等待服务恢复期间不能把已扫描到结尾的任务导出为完整录像', () => expect(recordingBlockReason({ ...task, status: 'waiting_service' } as unknown as Task)).not.toBe(''));
});
