import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import { ReviewPanel } from './components/ReviewPanel';
import type { Task } from './types';

const position = { side_to_move: 'red', pieces: [{ id: 'r', color: 'red', kind: 'rook', square: { file: 0, rank: 9 } }] };
const task = { rules_enabled: true, replay_mode: 'actual', initial_position: position, start_s: 0, end_s: 4,
  moves: [{ index: 1, move: null, status: 'needs_review', reason: '系统候选待核对', evidence_ids: [],
    candidates: [{ index: 1, piece_id: 'r', color: 'red', from: { file: 0, rank: 9 }, to: { file: 0, rank: 8 }, captured_piece_id: null }],
    video_end_s: 1 }],
} as unknown as Task;

it('分析或提交忙碌时，确认按钮仍可点击并可给出反馈', () => {
  const html = renderToStaticMarkup(createElement(ReviewPanel, { task, index: 1, position: task.initial_position!,
    picked: null, busy: true, videoTime: 1, onSave: () => {}, onSavePosition: () => {},
    onClose: () => {}, onPreview: () => {} }));
  const button = html.match(/<button[^>]*class="primary full"[^>]*>/)?.[0];
  expect(button).toBeTruthy();
  expect(button).not.toContain('disabled');
});

it('系统候选会预填棋子及目标位置，无需先手动修改才能确认', () => {
  const html = renderToStaticMarkup(createElement(ReviewPanel, { task, index: 1, position: task.initial_position!,
    picked: null, busy: false, videoTime: 1, onSave: () => {}, onSavePosition: () => {},
    onClose: () => {}, onPreview: () => {} }));
  expect(html).toContain('已选红车 → 第1列第9行');
  expect(html).toContain('确认并保存这一步');
});
