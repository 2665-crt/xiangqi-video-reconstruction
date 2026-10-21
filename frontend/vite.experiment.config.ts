import { mergeConfig } from 'vite';
import base from './vite.config';

const frontendPort = Number(process.env.EXPERIMENT_FRONTEND_PORT ?? 5281);
const backendPort = Number(process.env.EXPERIMENT_BACKEND_PORT ?? 8281);

export default mergeConfig(base, {
  server: { port: frontendPort, proxy: { '/api': `http://127.0.0.1:${backendPort}` } },
  preview: { port: frontendPort },
});
