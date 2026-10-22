import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import { RecordingFeedback } from './components/RecordingPanel';
import type { RecordingJob } from './types';

it('后台异步导出失败时，直接显示失败原因', () => {
  const job = { id: 'export-job', status: 'failed', error: '编码器返回错误', revision: 3 } as RecordingJob;
  const html = renderToStaticMarkup(createElement(RecordingFeedback, { job, revision: 3, error: '' }));
  expect(html).toContain('role="alert"');
  expect(html).toContain('编码器返回错误');
});

it('旧棋局录像链接明确标识版本过期', () => {
  const job = { id: 'export-job', status: 'completed', error: null, revision: 3 } as RecordingJob;
  const html = renderToStaticMarkup(createElement(RecordingFeedback, { job, revision: 4, error: '' }));
  expect(html).toContain('旧版本录像');
  expect(html).toContain('请重新导出');
  expect(html).toContain('/api/recordings/export-job/video');
});

it('当前版本录像提供查看入口，请求失败也有反馈', () => {
  const job = { id: 'export-job', status: 'completed', error: null, revision: 4 } as RecordingJob;
  const html = renderToStaticMarkup(createElement(RecordingFeedback, { job, revision: 4, error: '服务暂不可用' }));
  expect(html).toContain('服务暂不可用');
  expect(html).toContain('查看录像');
  expect(html).not.toContain('旧版本录像');
});
