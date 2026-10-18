import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import { ReviewPanel } from './components/ReviewPanel';
import type { Task } from './types';

const position = { side_to_move: 'red', pieces: [{ id: 'r', color: 'red', kind: 'rook', square: { file: 0, rank: 9 } }] };
const task = { rules_enabled: true, replay_mode: 'actual', initial_position: position, start_s: 0, end_s: 4,
  metadata: { width: 901, height: 1001, duration_s: 10, codec: 'h264' },
  moves: [{ index: 1, move: null, status: 'needs_review', reason: '系统候选待核对', evidence_ids: [],
    candidates: [{ index: 1, piece_id: 'r', color: 'red', from: { file: 0, rank: 9 }, to: { file: 0, rank: 8 }, captured_piece_id: null }],
    video_start_s: 0, video_end_s: 1 }],
} as unknown as Task;

it('分析或提交忙碌时，确认按钮仍可点击并可给出反馈', () => {
  const html = renderToStaticMarkup(createElement(ReviewPanel, { task, index: 1, position: task.initial_position!,
    picked: null, busy: true, videoTime: 1, onSave: () => {}, onSavePosition: () => {},
    onClose: () => {}, onPreview: () => {} }));
  const button = html.match(/<button[^>]*class="primary full"[^>]*>/)?.[0];
  expect(button).toBeTruthy();
  expect(button).not.toContain('disabled');
});

it('审核显示整盘差异与原视频证据时间，并允许选区外真实落定', () => {
  const tail = { ...task, evidence_end_s: 6, moves: [{ ...task.moves[0], action_end_s: 5,
    evidence_ids: ['before.jpg', 'after.jpg'], evidence_times_s: [1, 5.1],
    board_check: { status: 'conflict' as const, visible_before: 90, visible_after: 88,
      before_mismatches: [], after_mismatches: [{ file: 8, rank: 0 }], unknown_before: [],
      unknown_after: [{ file: 0, rank: 8 }], reason: '整盘额外变化，需核对中间操作', backfilled: false } }] };
  const html = renderToStaticMarkup(createElement(ReviewPanel, { task: tail, index: 1, position: task.initial_position!,
    picked: null, busy: false, videoTime: 1, onSave: () => {}, onSavePosition: () => {}, onClose: () => {}, onPreview: () => {} }));
  expect(html).toContain('整盘额外变化');
  expect(html).toContain('操作后差异：9,1');
  expect(html).toContain('5.100 秒');
  expect(html).toContain('选区外 5.000 秒核对');
  expect(html).toContain('max="6"');
});

it('系统候选会预填棋子及目标位置，无需先手动修改才能确认', () => {
  const html = renderToStaticMarkup(createElement(ReviewPanel, { task, index: 1, position: task.initial_position!,
    picked: null, busy: false, videoTime: 1, onSave: () => {}, onSavePosition: () => {},
    onClose: () => {}, onPreview: () => {} }));
  expect(html).toContain('已选红车 → 第1列第9行');
  expect(html).toContain('确认并保存这一步');
});
