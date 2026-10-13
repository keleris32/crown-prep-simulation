import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset paths: the built site works from any address or sub-folder
  // (e.g. https://user.github.io/tooth-prep/), not only from the root of a domain.
  base: "./",
});
