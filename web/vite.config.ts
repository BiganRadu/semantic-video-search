import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API is proxied in development so the app talks to one origin, exactly as
// it will in production where the Go server serves web/dist itself.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8080", changeOrigin: true },
    },
  },
  build: { outDir: "dist" },
});
