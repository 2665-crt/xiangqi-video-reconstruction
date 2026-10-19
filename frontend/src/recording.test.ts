import { describe, expect, it } from 'vitest';
import { recordingBlockReason } from './components/RecordingPanel';
import type { Task } from './types';

const task = { status: 'ready', start_s: 10, end_s: 120, processed_video_s: 120,
  initial_position: { side_to_move: 'red', pieces: [{ id:'r', color:'red', kind:'rook', square:{file:0,rank:9} }] },
  moves: [{ index: 1, status: 'auto_verified', video_start_s:10, video_end_s:60,
    move:{piece_id:'r', color:'red',from:{file:0,rank:9},to:{file:0,rank:8}}, position_after: { pieces: [{ id:'r', color:'red', kind:'rook', square:{file:0,rank:8} }] } }],
} as unknown as Task;
describe('录像导出入口', () => {
  it('完整确认的选区可以导出', () => expect(recordingBlockReason(task)).toBe(''));
  it.each(['needs_review', 'tentative'] as const)('已有走法为 %s 时仍可导出', status => expect(recordingBlockReason({ ...task, moves: [{ ...task.moves[0], status }] })).toBe(''));
  it('纯疑点不影响其他已有步骤导出', () => expect(recordingBlockReason({ ...task, moves: [...task.moves, { ...task.moves[0], index:2, move:null, position_after:null, status:'needs_review', change_type:'review' }] })).toBe(''));
  it('只存在疑点而没有走棋时不生成空录像', () => expect(recordingBlockReason({ ...task, moves: [{ ...task.moves[0], move:null, position_after:null, status:'needs_review', change_type:'review' }] })).toContain('步骤'));
  it('未分析到选区末尾也可导出已有步骤', () => expect(recordingBlockReason({ ...task, processed_video_s: 70 })).toBe(''));
  it('未确认的起始候选不能作为录像开局', () => expect(recordingBlockReason({ ...task, initial_position: null })).toContain('初始'));
  it.each(['analyzing', 'waiting_service', 'action_required', 'failed', 'interrupted'] as const)('%s 期间仍可导出已有步骤', status => expect(recordingBlockReason({ ...task, status })).toBe(''));
  it('未确认走法没有保存落子局面时仍可导出', () => expect(recordingBlockReason({ ...task, moves:[{ ...task.moves[0], status:'needs_review', position_after:null }] })).toBe(''));
  it('保留已保存的删子摆局录像，即使它不计入走棋数', () => expect(recordingBlockReason({ ...task, moves:[{ ...task.moves[0], move:null, status:'corrected', change_type:'setup', manual_override:true, position_after:{ pieces:[],side_to_move:'red' } }] })).toBe(''));
});
