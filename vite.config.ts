import { defineConfig } from "vite";

/**
 * The site is published as a GitHub Pages project site, so every asset is served
 * from /<repository>/ rather than from the domain root. `base` must match the
 * repository name or all bundle and worker URLs resolve to 404.
 */
export default defineConfig({
  base: process.env.BASE_PATH ?? "/depth-realtime/",
  build: {
    target: "es2022",
    sourcemap: true,
  },
  worker: {
    format: "es",
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
  },
});
