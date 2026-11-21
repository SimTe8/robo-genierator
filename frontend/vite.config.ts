import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // FastAPI backend (uvicorn app:app --app-dir backend --port 8000)
    proxy: {
      '/api': 'http://localhost:8000',
      '/outputs': 'http://localhost:8000',
    },
  },
})
