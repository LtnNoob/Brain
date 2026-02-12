import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3019,
    host: '0.0.0.0',
    open: false,
    proxy: {
      '/api': 'http://localhost:8019',
      '/ws': {
        target: 'ws://localhost:8019',
        ws: true,
      },
    },
  },
});
