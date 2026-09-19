import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { Stage, mono } from "../components/Stage";
import { Narration } from "../components/Narration";

export const Monitor = () => {
  const frame = useCurrentFrame();
  return (
    <Stage chapter="04 / RUNTIME MONITORING">
      <Interactive.Div
        name="Async monitoring heading"
        style={{
          position: "absolute",
          left: 80,
          top: 220,
          fontSize: 95,
          fontWeight: 800,
          lineHeight: 1.07,
          letterSpacing: -5,
          opacity: interpolate(frame, [8, 24], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Keep watching.
        <br />
        <span style={{ color: "#D6FA76" }}>As the agent runs.</span>
      </Interactive.Div>
      <div
        style={{
          position: "absolute",
          left: 80,
          top: 526,
          width: 920,
          height: 126,
          borderRadius: 22,
          background: "#253641",
          border: "1px solid #5F7681",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-around",
          fontSize: 45,
          fontWeight: 700,
        }}
      >
        <span>Customer</span>
        <span style={{ color: "#9EC9B6", fontSize: 55 }}>↔</span>
        <span>Your agent</span>
      </div>
      <svg
        style={{ position: "absolute", left: 80, top: 651 }}
        width="920"
        height="194"
      >
        <path
          d="M676 0V92Q676 112 656 112H470Q450 112 450 132V194"
          fill="none"
          stroke="#688976"
          strokeWidth="3"
          strokeDasharray="8 10"
        />
      </svg>
      <Interactive.Div
        name="Flowing event"
        style={{
          position: "absolute",
          left: 746,
          top: 662,
          width: 20,
          height: 20,
          background: "#D6FA76",
          boxShadow: "0 0 25px #D6FA76",
          borderRadius: "50%",
          translate: interpolate(
            frame,
            [40, 70, 100, 120],
            ["0px 0px", "0px 91px", "-226px 91px", "-226px 180px"],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
          ),
          opacity: interpolate(frame, [36, 45, 124, 136], [0, 1, 1, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      />
      <Interactive.Div
        name="Event capture description"
        style={{
          position: "absolute",
          left: 90,
          top: 712,
          fontFamily: mono,
          fontSize: 30,
          lineHeight: 1.5,
          color: "#ADC2CC",
          opacity: interpolate(frame, [47, 67], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Messages + context
        <br />+ tool activity
      </Interactive.Div>
      <Interactive.Div
        name="Litmus runtime receiver"
        style={{
          position: "absolute",
          left: 80,
          right: 80,
          top: 864,
          borderRadius: 24,
          background: "#D6FA76",
          color: "#152530",
          padding: "30px 36px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          opacity: interpolate(frame, [90, 111], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <div style={{ fontSize: 51, fontWeight: 800, letterSpacing: -2 }}>
          LitmusAI Runtime
        </div>
        <div
          style={{
            border: "2px solid #567444",
            borderRadius: 30,
            padding: "11px 18px",
            fontFamily: mono,
            fontSize: 24,
          }}
        >
          ASYNC
        </div>
      </Interactive.Div>
      <div
        style={{
          position: "absolute",
          left: 80,
          top: 1047,
          fontSize: 28,
          color: "#AEC3CC",
        }}
      >
        Connect your agent with event hooks and tool wrappers.
      </div>
      <Narration index={3} />
    </Stage>
  );
};
