import { describe, expect, it } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Board } from './components/Board';
import type { Calibration, Position } from './types';

const position: Position = { side_to_move: 'red', pieces: [
  { id: 'black-king', color: 'black', kind: 'king', square: { file: 4, rank: 0 } },
  { id: 'red-rook', color: 'red', kind: 'rook', square: { file: 0, rank: 9 } },
] };

describe('视频方向与虚拟棋盘一致', () => {
  it.each([
    ['normal', 'translate(50 473)', 'translate(238 50)'],
    ['rotated_180', 'translate(426 50)', 'translate(238 473)'],
    ['mirrored', 'translate(426 473)', 'translate(238 50)'],
    ['mirrored_rotated_180', 'translate(50 50)', 'translate(238 473)'],
  ] as const)('%s 保留棋子身份并变换位置', (orientation, rook, king) => {
    const props = { position, orientation: orientation as Calibration['orientation'] };
    const html = renderToStaticMarkup(createElement(Board, props));
    expect(html).toContain(`data-piece-id="red-rook" transform="${rook}"`);
    expect(html).toContain(`data-piece-id="black-king" transform="${king}"`);
    expect(html).toContain('>将</text>');
    expect(html).toContain('>车</text>');
    expect(position.pieces[0].color).toBe('black');
  });
});
