import { Easing, Interactive, interpolate, useCurrentFrame } from "remotion";
import { Stage, mono } from "../components/Stage";
import { Narration } from "../components/Narration";

export const Support = () => {
  const frame = useCurrentFrame();
  return (
    <Stage chapter="02 / IN PRODUCTION">
      <Interactive.Div
        name="Normal conversation heading"
        style={{
          position: "absolute",
          left: 80,
          top: 225,
          fontSize: 94,
          fontWeight: 800,
          lineHeight: 1.08,
          letterSpacing: -4.8,
          opacity: interpolate(frame, [8, 24], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        A normal day.
        <br />
        <span style={{ color: "#D6FA76" }}>A helpful agent.</span>
      </Interactive.Div>
      <Interactive.Div
        name="Customer invoice request"
        style={{
          position: "absolute",
          left: 80,
          right: 135,
          top: 525,
          padding: "34px 40px",
          background: "#243540",
          border: "1px solid #49606C",
          borderRadius: "26px 26px 26px 6px",
          opacity: interpolate(frame, [22, 38], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          translate: interpolate(frame, [22, 45], ["0px 35px", "0px 0px"], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          }),
        }}
      >
        <div
          style={{
            fontFamily: mono,
            fontSize: 25,
            color: "#AEC3CC",
            marginBottom: 20,
          }}
        >
          CUSTOMER
        </div>
        <div style={{ fontSize: 54, lineHeight: 1.25, fontWeight: 600 }}>
          Can you resend
          <br />
          my invoice?
        </div>
      </Interactive.Div>
      <Interactive.Div
        name="Agent response"
        style={{
          position: "absolute",
          left: 185,
          right: 80,
          top: 822,
          padding: "34px 40px",
          background: "#D6FA76",
          color: "#152530",
          borderRadius: "26px 26px 6px 26px",
          opacity: interpolate(frame, [80, 100], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          translate: interpolate(frame, [80, 103], ["0px 35px", "0px 0px"], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          }),
        }}
      >
        <div style={{ fontFamily: mono, fontSize: 25, marginBottom: 18 }}>
          SUPPORT AGENT
        </div>
        <div style={{ fontSize: 50, fontWeight: 700 }}>
          I can help with that.
        </div>
      </Interactive.Div>
      <Narration index={1} />
    </Stage>
  );
};
