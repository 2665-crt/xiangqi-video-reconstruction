import { describe, expect, it } from 'vitest';
import { gridPoint } from './calibration';

describe('棋盘网格透视', () => {
  it('四角与确认范围一致', () => {
    const corners = [{ x: 50, y: 20 }, { x: 150, y: 20 }, { x: 200, y: 120 }, { x: 0, y: 120 }];
    expect(gridPoint(corners, 0, 0)).toEqual(corners[0]);
    expect(gridPoint(corners, 1, 0)).toEqual(corners[1]);
    expect(gridPoint(corners, 1, 1)).toEqual(corners[2]);
    expect(gridPoint(corners, 0, 1)).toEqual(corners[3]);
  });
  it('梯形中央交叉点保持透视而非简单线性插值', () => {
    expect(gridPoint([{ x: 50, y: 0 }, { x: 150, y: 0 }, { x: 200, y: 100 }, { x: 0, y: 100 }], .5, .5))
      .toEqual({ x: 100, y: 100 / 3 });
  });
});
