import { Easing, Interactive, interpolate, useCurrentFrame } from "remotion";
import { Stage, mono } from "../components/Stage";
import { Narration } from "../components/Narration";

export const Tests = () => {
  const frame = useCurrentFrame();
  return (
    <Stage light chapter="01 / BEFORE DEPLOYMENT">
      <Interactive.Div
        name="Opening hook"
        style={{
          position: "absolute",
          left: 80,
          top: 232,
          fontSize: 104,
          fontWeight: 800,
          lineHeight: 1.04,
          letterSpacing: -6,
          opacity: interpolate(frame, [0, 12], [0, 1], {
            extrapolateRight: "clamp",
          }),
          translate: interpolate(frame, [0, 24], ["0px 36px", "0px 0px"], {
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          }),
        }}
      >
        Your agent
        <br />
        passed the tests.
      </Interactive.Div>
      <Interactive.Div
        name="Test pass stamp"
        style={{
          position: "absolute",
          left: 80,
          top: 542,
          width: 920,
          height: 230,
          background: "#E2E9D4",
          border: "2px solid #CBDBBA",
          borderRadius: 24,
          display: "flex",
          alignItems: "center",
          padding: 46,
          gap: 34,
          scale: interpolate(frame, [15, 33], [0.95, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          }),
          opacity: interpolate(frame, [15, 30], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <svg width="86" height="86" viewBox="0 0 86 86">
          <circle cx="43" cy="43" r="42" fill="#244E42" />
          <path
            d="M23 44L37 58L64 29"
            fill="none"
            stroke="#D6FA76"
            strokeWidth="7"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <div>
          <div
            style={{
              fontFamily: mono,
              fontSize: 27,
              letterSpacing: 3,
              marginBottom: 10,
            }}
          >
            PREDEPLOYMENT CHECKS
          </div>
          <div style={{ fontSize: 72, fontWeight: 800, letterSpacing: -3 }}>
            PASS
          </div>
        </div>
      </Interactive.Div>
      <Interactive.Div
        name="The question"
        style={{
          position: "absolute",
          left: 80,
          top: 870,
          fontSize: 75,
          fontWeight: 700,
          letterSpacing: -3,
          opacity: interpolate(frame, [104, 128], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          translate: interpolate(frame, [104, 128], ["0px 24px", "0px 0px"], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        What happens next?
      </Interactive.Div>
      <div
        style={{
          position: "absolute",
          left: 80,
          top: 1038,
          fontSize: 24,
          fontFamily: mono,
          opacity: 0.55,
        }}
      >
        AN ILLUSTRATED AGENT STORY
      </div>
      <Narration index={0} light />
    </Stage>
  );
};
