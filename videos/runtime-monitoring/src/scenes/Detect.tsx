import { Easing, Interactive, interpolate, useCurrentFrame } from "remotion";
import { Stage, mono } from "../components/Stage";
import { Narration } from "../components/Narration";

export const Detect = () => {
  const frame = useCurrentFrame();
  return (
    <Stage chapter="05 / POLICY MATCH">
      <Interactive.Div
        name="Detection heading"
        style={{
          position: "absolute",
          left: 80,
          top: 218,
          fontSize: 96,
          fontWeight: 800,
          lineHeight: 1.07,
          letterSpacing: -5,
          opacity: interpolate(frame, [4, 20], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        A rule fires.
        <br />
        <span style={{ color: "#FF8871" }}>An event follows.</span>
      </Interactive.Div>
      <Interactive.Div
        name="Evidence receipt"
        style={{
          position: "absolute",
          left: 80,
          top: 495,
          width: 920,
          height: 570,
          background: "#F4F1E8",
          color: "#14232C",
          borderRadius: 22,
          overflow: "hidden",
          opacity: interpolate(frame, [34, 54], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          translate: interpolate(frame, [34, 61], ["0px 45px", "0px 0px"], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          }),
        }}
      >
        <div
          style={{
            background: "#FF8871",
            height: 83,
            display: "flex",
            alignItems: "center",
            padding: "0 38px",
            fontFamily: mono,
            fontSize: 28,
            fontWeight: 500,
            letterSpacing: 2,
          }}
        >
          RISK DETECTED
        </div>
        <div style={{ padding: "30px 38px" }}>
          <div
            style={{
              fontSize: 49,
              fontWeight: 800,
              letterSpacing: -2,
              marginBottom: 30,
            }}
          >
            Forbidden destination
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "260px 1fr",
              rowGap: 18,
              fontSize: 32,
              fontFamily: mono,
            }}
          >
            <span style={{ opacity: 0.56 }}>conversation</span>
            <span>support-042</span>
            <span style={{ opacity: 0.56 }}>policy</span>
            <span>support-tools v1</span>
            <span style={{ opacity: 0.56 }}>stage</span>
            <span>requested</span>
            <span style={{ opacity: 0.56 }}>evidence</span>
            <span>tool.requested</span>
          </div>
          <div
            style={{
              borderTop: "1px dashed #9DA59F",
              marginTop: 30,
              paddingTop: 22,
              fontSize: 29,
              fontFamily: mono,
              color: "#476457",
            }}
          >
            com.litmusai.threat.detected
          </div>
        </div>
      </Interactive.Div>
      <Narration index={4} />
    </Stage>
  );
};
