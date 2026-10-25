import { expect, it } from 'vitest';
import { api } from './api';

it('超过 5 GiB 的文件在传输前提示用户', async () => {
  await expect(api.upload({ size: 5 * 1024 ** 3 + 1 } as File)).rejects.toThrow('5 GiB');
});
