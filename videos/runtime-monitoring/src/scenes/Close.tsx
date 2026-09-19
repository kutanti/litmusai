import { Easing, Interactive, interpolate, useCurrentFrame } from "remotion";
import { Stage, mono } from "../components/Stage";
import { Narration } from "../components/Narration";

export const Close = () => {
  const frame = useCurrentFrame();
  return (
    <Stage light chapter="TEST + MONITOR">
      <Interactive.Div
        name="Closing promise"
        style={{
          position: "absolute",
          left: 80,
          top: 242,
          fontSize: 117,
          fontWeight: 800,
          lineHeight: 1.07,
          letterSpacing: -6.5,
          opacity: interpolate(frame, [7, 24], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          translate: interpolate(frame, [7, 30], ["0px 30px", "0px 0px"], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          }),
        }}
      >
        Test before.
        <br />
        <span style={{ color: "#32665A" }}>Monitor after.</span>
      </Interactive.Div>
      <Interactive.Div
        name="Litmus strips"
        style={{
          position: "absolute",
          left: 87,
          top: 623,
          display: "flex",
          gap: 18,
          rotate: "-9deg",
          scale: interpolate(frame, [29, 56], [0.9, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          }),
          opacity: interpolate(frame, [29, 45], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <div
          style={{
            width: 49,
            height: 194,
            borderRadius: 10,
            background: "#32665A",
          }}
        />
        <div
          style={{
            width: 49,
            height: 194,
            borderRadius: 10,
            background: "#85D8BA",
          }}
        />
        <div
          style={{
            width: 49,
            height: 194,
            borderRadius: 10,
            background: "#FF8871",
          }}
        />
      </Interactive.Div>
      <div style={{ position: "absolute", left: 322, top: 638 }}>
        <div style={{ fontSize: 74, fontWeight: 800, letterSpacing: -3.5 }}>
          LitmusAI
        </div>
        <div
          style={{
            marginTop: 17,
            fontSize: 34,
            lineHeight: 1.32,
            color: "#52655C",
          }}
        >
          Testing and runtime monitoring
          <br />
          for AI agents.
        </div>
      </div>
      <Interactive.Div
        name="Repository call to action"
        style={{
          position: "absolute",
          left: 80,
          right: 80,
          top: 918,
          height: 100,
          background: "#14232C",
          color: "#D6FA76",
          borderRadius: 18,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: mono,
          fontSize: 36,
          opacity: interpolate(frame, [76, 94], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        github.com/kutanti/litmusai ↗
      </Interactive.Div>
      <div
        style={{
          position: "absolute",
          left: 80,
          top: 1052,
          fontSize: 28,
          fontFamily: mono,
          color: "#52655C",
        }}
      >
        EXPERIMENTAL RUNTIME · AVAILABLE ON MAIN
      </div>
      <Narration index={7} light />
    </Stage>
  );
};
