import { createTikTokStyleCaptions, type Caption } from "@remotion/captions";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { fontFamily } from "./Stage";

export const Captions: React.FC<{ captions: Caption[]; light?: boolean }> = ({
  captions,
  light = false,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const time = (frame / fps) * 1000;
  const { pages } = createTikTokStyleCaptions({
    captions,
    combineTokensWithinMilliseconds: 1450,
  });
  const page = pages.find(
    (p, i) =>
      time >= p.startMs &&
      time <
        Math.min(
          p.startMs + p.durationMs + 180,
          pages[i + 1]?.startMs ?? Infinity,
        ),
  );
  if (!page) return null;
  return (
    <div
      style={{
        position: "absolute",
        left: 78,
        right: 78,
        top: 1150,
        minHeight: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily,
        fontSize: 43,
        fontWeight: 600,
        lineHeight: 1.32,
        textAlign: "center",
        color: light ? "#152530" : "#F4F1E8",
      }}
    >
      <div>
        {page.tokens.map((token) => (
          <span
            key={token.fromMs}
            style={{
              color:
                token.fromMs <= time && token.toMs > time
                  ? light
                    ? "#357263"
                    : "#D6FA76"
                  : undefined,
            }}
          >
            {token.text}
          </span>
        ))}
      </div>
    </div>
  );
};
