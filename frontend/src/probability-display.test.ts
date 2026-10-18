import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test } from 'vitest';
import { Board } from './components/Board';
import type { Position } from './types';

test('largest-probability piece is visible and its probability stays off the board', () => {
  const position: Position = { pieces: [{ id: 'candidate', color: 'red', kind: 'rook', square: { file: 4, rank: 3 } }], side_to_move: 'red' };
  const html = renderToStaticMarkup(createElement(Board, { position, step: 0, reviews: [{ square: { file: 4, rank: 3 }, color: 'red', prediction: 'rook', score: .64, reason: '已显示最大概率棋种' }] }));
  const text = html.replace(/<[^>]*>/g, '');
  expect(text).toContain('车');
  expect(text).not.toMatch(/64%|16%|20%|0\.64/);
  expect(html).not.toContain('>?<');
});
