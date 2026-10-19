import { expect, it } from 'vitest';
import { visibleRecords, displayStep, nextStep, reviewIntervals } from './replay';
import type { Task, MoveRecord } from './types';

const before = { pieces: [{ id:'r', color:'red', kind:'rook', square:{file:0,rank:9} }], side_to_move:'red' };
const after = { ...before, pieces:[{ ...before.pieces[0], square:{file:0,rank:8} }] };
const motion = { index:1, status:'auto_verified', change_type:'move', move:{index:1,piece_id:'r',color:'red',from:{file:0,rank:9},to:{file:0,rank:8},captured_piece_id:null},
  position_before:before, position_after:after, evidence_ids:['before.jpg','after.jpg'], candidates:[], video_start_s:0, video_end_s:1, reason:'' } as MoveRecord;
const empty = { ...motion, index:2, status:'needs_review', change_type:'review', move:null, position_after:null, observed_positions:[], candidates:[], reason:'无落点候选' } as MoveRecord;

it('空记录不进入步骤或编号，保留原始数据供审核', () => {
  const task = { moves:[motion, ...Array.from({length:1500},(_,i)=>({...empty,index:i+2}))], initial_position:before } as Task;
  expect(visibleRecords(task)).toEqual([motion]);
  expect(displayStep(task,1501)).toBe(1);
  expect(task.moves).toHaveLength(1501);
  expect(reviewIntervals(task)).toHaveLength(1);
  expect(reviewIntervals(task)[0].records).toHaveLength(1500);
});

it('整盘证据一致但规则待审核的真实移动保留，冲突候选不冒充走棋', () => {
  const pending = { ...empty, candidates:[motion.move!], board_check:{status:'verified'} } as MoveRecord;
  expect(visibleRecords({moves:[pending]} as Task)).toEqual([pending]);
  pending.board_check!.status='conflict';
  expect(visibleRecords({moves:[pending]} as Task)).toEqual([]);
});

it('有直接走法的待核对移动仍显示，同格无移动排除', () => {
  const pending = { ...motion,index:2,status:'needs_review' } as MoveRecord;
  const stationary = { ...motion,index:3,move:{...motion.move!,to:motion.move!.from} };
  expect(visibleRecords({moves:[motion,pending,stationary]} as Task)).toEqual([motion,pending]);
});

it('保留的走法没有使棋子实际改变位置时不计步', () => {
  const stationary={...motion,position_after:before};
  expect(visibleRecords({moves:[stationary]} as Task)).toEqual([]);
});

it('空记录隐藏后仍不能自动跨越未解决缺口', () => {
  const later={...motion,index:3};
  const task={moves:[motion,empty,later],replay_mode:'actual'} as Task;
  expect(nextStep(task,1)).toBeNull();
});
