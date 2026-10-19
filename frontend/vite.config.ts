import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: { host: '127.0.0.1', port: 5278, strictPort: true, proxy: { '/api': 'http://127.0.0.1:8278' } },
  preview: { host: '127.0.0.1', port: 5278, strictPort: true },
  test: { include: ['src/**/*.test.ts'] },
});
