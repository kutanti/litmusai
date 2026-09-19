import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createTikTokStyleCaptions } from "@remotion/captions";

const project = new URL("../", import.meta.url);
const timeline = JSON.parse(
  readFileSync(new URL("src/timeline.json", project), "utf8"),
);
const out = new URL("out/", project);
mkdirSync(out, { recursive: true });

const timestamp = (ms) => {
  const total = Math.round(ms);
  const hh = Math.floor(total / 3600000);
  const mm = Math.floor(total / 60000) % 60;
  const ss = Math.floor(total / 1000) % 60;
  return (
    [hh, mm, ss].map((n) => String(n).padStart(2, "0")).join(":") +
    "," +
    String(total % 1000).padStart(3, "0")
  );
};

const entries = [];
for (const scene of timeline) {
  const captions = JSON.parse(
    readFileSync(
      new URL("public/voiceover/" + scene.id + ".json", project),
      "utf8",
    ),
  );
  const { pages } = createTikTokStyleCaptions({
    captions,
    combineTokensWithinMilliseconds: 1450,
  });
  const offsetMs = ((scene.startFrame + 12) / 30) * 1000;
  for (const [i, page] of pages.entries()) {
    const end = Math.min(
      page.startMs + page.durationMs + 180,
      pages[i + 1]?.startMs ?? Infinity,
    );
    entries.push(
      entries.length +
        1 +
        "\n" +
        timestamp(offsetMs + page.startMs) +
        " --> " +
        timestamp(offsetMs + end) +
        "\n" +
        page.text.trim() +
        "\n",
    );
  }
}
const target = new URL("litmusai-runtime-story.srt", out);
writeFileSync(target, entries.join("\n"));
console.log(fileURLToPath(target));
