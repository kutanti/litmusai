import { loadFont } from "@remotion/google-fonts/Manrope";
import { loadFont as loadMono } from "@remotion/google-fonts/IBMPlexMono";
import {
  AbsoluteFill,
  Interactive,
  interpolate,
  useCurrentFrame,
} from "remotion";
import type { PropsWithChildren } from "react";

export const { fontFamily } = loadFont("normal", {
  weights: ["500", "600", "700", "800"],
  subsets: ["latin"],
});
export const { fontFamily: mono } = loadMono("normal", {
  weights: ["400", "500"],
  subsets: ["latin"],
});

export const Stage: React.FC<
  PropsWithChildren<{ light?: boolean; chapter: string }>
> = ({ children, light = false, chapter }) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill
      style={{
        fontFamily,
        background: light ? "#F4F1E8" : "#101D27",
        color: light ? "#14232C" : "#F4F1E8",
        overflow: "hidden",
      }}
    >
      <AbsoluteFill
        style={{
          backgroundImage: light
            ? "radial-gradient(#15253012 1px, transparent 1px)"
            : "radial-gradient(#DAE8ED13 1px, transparent 1px)",
          backgroundSize: "30px 30px",
          opacity: 0.8,
        }}
      />
      <Interactive.Div
        name="Slow orbit"
        style={{
          position: "absolute",
          left: 610,
          top: 440,
          width: 820,
          height: 820,
          border: light ? "1px solid #14232C12" : "1px solid #D6FA761A",
          borderRadius: "50%",
          scale: interpolate(frame, [0, 260], [0.8, 1.15], {
            extrapolateRight: "clamp",
          }),
        }}
      />
      <div
        style={{
          position: "absolute",
          top: 76,
          left: 80,
          display: "flex",
          alignItems: "center",
          gap: 17,
        }}
      >
        <div style={{ display: "flex", gap: 5, rotate: "-14deg" }}>
          <div
            style={{
              width: 10,
              height: 34,
              borderRadius: 3,
              background: light ? "#32665A" : "#D6FA76",
            }}
          />
          <div
            style={{
              width: 10,
              height: 34,
              borderRadius: 3,
              background: "#85D8BA",
            }}
          />
          <div
            style={{
              width: 10,
              height: 34,
              borderRadius: 3,
              background: "#FF8871",
            }}
          />
        </div>
        <span style={{ fontSize: 37, fontWeight: 800, letterSpacing: -1.4 }}>
          LitmusAI
        </span>
      </div>
      <div
        style={{
          position: "absolute",
          right: 80,
          top: 91,
          fontFamily: mono,
          fontSize: 20,
          letterSpacing: 1.8,
          opacity: 0.58,
        }}
      >
        {chapter}
      </div>
      {children}
    </AbsoluteFill>
  );
};
