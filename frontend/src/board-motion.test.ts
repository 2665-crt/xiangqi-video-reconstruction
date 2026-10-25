import { describe, expect, it } from 'vitest';
import { planBoardTransition } from './components/Board';
import { displayStep } from './replay';
import type { Position, Task } from './types';

const before: Position = { side_to_move: 'red', pieces: [
  { id: 'r', color: 'red', kind: 'rook', square: { file: 0, rank: 9 } },
  { id: 'p', color: 'black', kind: 'pawn', square: { file: 0, rank: 5 } },
] };
const after: Position = { side_to_move: 'black', pieces: [
  { ...before.pieces[0], square: { file: 0, rank: 5 } },
] };

describe('虚拟棋盘动画', () => {
  it('局面相同时切换方向仍替换棋盘并取消旧坐标动画', () => {
    expect(planBoardTransition(before, before, 0, false, true).type).toBe('replace');
  });
  it('单步吃子保留移动起终点和被吃棋子，供落子阶段移除', () => {
    const transition = planBoardTransition(before, after, 1);
    expect(transition.type).toBe('move');
    expect(transition.moved[0].before.square).toEqual({ file: 0, rank: 9 });
    expect(transition.moved[0].after.square).toEqual({ file: 0, rank: 5 });
    expect(transition.removed.map(p => p.id)).toEqual(['p']);
  });
  it('上一步反向移动，并恢复吃子前的棋子', () => {
    const transition = planBoardTransition(after, before, -1);
    expect(transition.type).toBe('move');
    expect(transition.moved[0].after.square).toEqual({ file: 0, rank: 9 });
    expect(transition.added.map(p => p.id)).toEqual(['p']);
  });
  it.each([1, 2])('相邻可见走棋间有 %i 条隐藏疑点时仍播放移动及反向吃子动画', hidden => {
    const action = { piece_id:'r', color:'red', from:before.pieces[0].square, to:after.pieces[0].square };
    const task = { initial_position:before, moves:[
      { index:1, status:'auto_verified', move:{ ...action, from:{file:0,rank:8}, to:before.pieces[0].square }, position_before:{ ...before, pieces:[{ ...before.pieces[0], square:{file:0,rank:8} },before.pieces[1]] }, position_after:before },
      ...Array.from({length:hidden}, (_,i) => ({index:i+2,status:'needs_review',change_type:'review',move:null,candidates:[]})),
      { index:hidden+2, status:'auto_verified', move:action, position_before:before, position_after:after },
    ] } as unknown as Task;
    const delta = displayStep(task, hidden+2)-displayStep(task, 1);
    expect(delta).toBe(1);
    expect(planBoardTransition(before,after,delta).type).toBe('move');
    expect(planBoardTransition(after,before,-delta).type).toBe('move');
  });
  it('跨多步即使只剩一枚棋子的位置不同，也使用局面过渡', () => {
    expect(planBoardTransition(before, after, 5).type).toBe('replace');
  });
  it('拖动视频到相邻一步也使用局面过渡', () => {
    expect(planBoardTransition(before, after, 1, true).type).toBe('replace');
  });
  it('局面快照或人工校正同时改变多个棋子时，不编造单步走棋', () => {
    const changed: Position = { ...before, pieces: before.pieces.map(p => ({ ...p, square: { ...p.square, file: 1 } })) };
    expect(planBoardTransition(before, changed, 1).type).toBe('replace');
  });
  it('视频重复时间更新和后端轮询不重播相同局面的动画', () => {
    expect(planBoardTransition(before, structuredClone(before), 0).type).toBe('none');
  });
  it('初次载入和身份不明确的棋子变化采用过渡', () => {
    expect(planBoardTransition(null, before, 0).type).toBe('replace');
    expect(planBoardTransition(before, { ...after, pieces: [{ ...after.pieces[0], id: 'unknown' }] }, 1).type).toBe('replace');
  });
});
