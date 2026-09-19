import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { Stage, mono } from "../components/Stage";
import { Narration } from "../components/Narration";

export const Policies = () => {
  const frame = useCurrentFrame();
  return (
    <Stage chapter="07 / CONFIGURE YOUR POLICIES">
      <Interactive.Div
        name="Policy heading"
        style={{
          position: "absolute",
          left: 80,
          top: 220,
          fontSize: 96,
          fontWeight: 800,
          lineHeight: 1.06,
          letterSpacing: -4.6,
          opacity: interpolate(frame, [8, 22], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Different agents.
        <br />
        <span style={{ color: "#D6FA76" }}>Your policies.</span>
      </Interactive.Div>
      <Interactive.Div
        name="Tool usage limit"
        style={{
          position: "absolute",
          left: 80,
          right: 80,
          top: 510,
          height: 154,
          borderBottom: "1px solid #445B66",
          opacity: interpolate(frame, [18, 35], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 30 }}>
          <div
            style={{
              fontFamily: mono,
              fontSize: 58,
              fontWeight: 500,
              color: "#FF8871",
              width: 150,
            }}
          >
            21<span style={{ fontSize: 26 }}>/20</span>
          </div>
          <div>
            <div style={{ fontSize: 47, fontWeight: 700 }}>Tool-use limits</div>
            <div style={{ fontSize: 30, color: "#AEC3CC", marginTop: 10 }}>
              Count requests across a conversation.
            </div>
          </div>
        </div>
      </Interactive.Div>
      <Interactive.Div
        name="Versioned conversation policies"
        style={{
          position: "absolute",
          left: 80,
          right: 80,
          top: 702,
          height: 154,
          borderBottom: "1px solid #445B66",
          opacity: interpolate(frame, [61, 80], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 30 }}>
          <div
            style={{
              fontFamily: mono,
              fontSize: 58,
              color: "#85D8BA",
              width: 150,
            }}
          >
            v1
          </div>
          <div>
            <div style={{ fontSize: 47, fontWeight: 700 }}>
              Conversation policies
            </div>
            <div style={{ fontSize: 30, color: "#AEC3CC", marginTop: 10 }}>
              Versioned rubrics. Captured evidence.
            </div>
          </div>
        </div>
      </Interactive.Div>
      <Interactive.Div
        name="Optional deeper evaluation"
        style={{
          position: "absolute",
          left: 80,
          right: 80,
          top: 895,
          opacity: interpolate(frame, [111, 131], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 30 }}>
          <div
            style={{
              fontFamily: mono,
              fontSize: 58,
              color: "#D6FA76",
              width: 150,
            }}
          >
            1→2
          </div>
          <div>
            <div style={{ fontSize: 47, fontWeight: 700 }}>
              Deeper evaluation
            </div>
            <div style={{ fontSize: 30, color: "#AEC3CC", marginTop: 10 }}>
              Optional. Your compatible evaluator.
            </div>
          </div>
        </div>
      </Interactive.Div>
      <Narration index={6} />
    </Stage>
  );
};
