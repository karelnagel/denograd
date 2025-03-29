import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const blog = defineCollection({
  loader: glob({ base: "./src/blog", pattern: "**/*.(md|mdx)", generateId: ({ entry }) => entry.split(".md")[0] }),
  schema: z.object({
    title: z.string(),
  }),
});

export const collections = { blog };
