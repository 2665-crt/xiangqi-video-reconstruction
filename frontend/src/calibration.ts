export function gridPoint(corners: { x: number; y: number }[], u: number, v: number) {
  const [p, q, r, s] = corners;
  const dx1 = q.x - r.x, dx2 = s.x - r.x, dx3 = p.x - q.x + r.x - s.x;
  const dy1 = q.y - r.y, dy2 = s.y - r.y, dy3 = p.y - q.y + r.y - s.y;
  const determinant = dx1 * dy2 - dx2 * dy1;
  const g = Math.abs(determinant) < 1e-8 ? 0 : (dx3 * dy2 - dx2 * dy3) / determinant;
  const h = Math.abs(determinant) < 1e-8 ? 0 : (dx1 * dy3 - dx3 * dy1) / determinant;
  const denominator = g * u + h * v + 1;
  return { x: ((q.x - p.x + g * q.x) * u + (s.x - p.x + h * s.x) * v + p.x) / denominator,
    y: ((q.y - p.y + g * q.y) * u + (s.y - p.y + h * s.y) * v + p.y) / denominator };
}
