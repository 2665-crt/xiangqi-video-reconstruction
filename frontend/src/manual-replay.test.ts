import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import { MoveList } from './components/MoveList';
import { ReviewPanel } from './components/ReviewPanel';
import { nextStep, positionBefore, stepAtVideoTime } from './replay';
import type { Task } from './types';

const initial = { pieces: [{ id: 'r', color: 'red', kind: 'rook', square: { file: 0, rank: 9 } }], side_to_move: 'red' };
const saved = { ...initial, pieces: [{ ...initial.pieces[0], square: { file: 0, rank: 8 } }] };
const task = { id: 'fixture', replay_mode: 'actual', rules_enabled: true, initial_position: initial,
  metadata: { width: 901, height: 1001, duration_s: 5, codec: 'h264' },
  start_s: 0, end_s: 5, confirmed_prefix_length: 3, moves: [
    { index: 1, move: null, status: 'corrected', change_type: 'setup', position_after: saved, video_end_s: 1, candidates: [], evidence_ids: [] },
    { index: 2, move: null, status: 'corrected', change_type: 'review', position_after: saved, video_end_s: 2, candidates: [], evidence_ids: [] },
    { index: 3, move: null, status: 'auto_verified', change_type: 'setup', position_before: initial, position_after: saved, video_end_s: 3, candidates: [], evidence_ids: [] },
  ] } as unknown as Task;

it('下一步跳过不计走棋的记录，编辑起点采用上一保存局面', () => {
  expect(nextStep(task, 1)).toBe(3);
  expect(positionBefore(task, 3)).toEqual(saved);
});

it('步骤列表隐藏忽略记录，显示连续的两手编号', () => {
  const html = renderToStaticMarkup(createElement(MoveList, { task, step: 3, onSelect: () => {} }));
  expect((html.match(/class="move-row/g) ?? []).length).toBe(2);
  expect(html).not.toContain('不计入走棋');
  expect(html).toContain('>02</span>');
  expect(html).not.toContain('class="move-index">03</span>');
});

it('单步编辑提供局面校正入口和点击移动，不显示坐标输入或连续补录', () => {
  const html = renderToStaticMarkup(createElement(ReviewPanel, { task, index: 1, position: task.initial_position!,
    picked: null, busy: false, videoTime: 1, onSave: () => {}, onSavePosition: () => {}, onClose: () => {}, onPreview: () => {} }));
  expect(html).toContain('局面校正');
  expect(html).not.toContain('aria-label="目标列"');
  expect(html).not.toContain('aria-label="目标行"');
  expect(html).not.toContain('加入本段下一手');
});

it('旧规则任务人工校正后，视频对照仍能定位前缀外的已知局面', () => {
  const legacy = { ...task, replay_mode: 'legal', confirmed_prefix_length: 1,
    moves: [{ ...task.moves[0], manual_override: true }, { ...task.moves[1], status: 'needs_review', position_after: null }, task.moves[2]],
  } as Task;
  expect(stepAtVideoTime(legacy, 4)).toBe(3);
});
