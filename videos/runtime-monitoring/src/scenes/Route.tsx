import { Easing, Interactive, interpolate, useCurrentFrame } from "remotion";
import { Stage, mono } from "../components/Stage";
import { Narration } from "../components/Narration";

export const Route = () => {
  const frame = useCurrentFrame();
  return (
    <Stage chapter="06 / YOUR RESPONSE">
      <Interactive.Div
        name="Routing heading"
        style={{
          position: "absolute",
          left: 80,
          top: 220,
          fontSize: 98,
          fontWeight: 800,
          lineHeight: 1.07,
          letterSpacing: -5,
          opacity: interpolate(frame, [8, 24], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Your event system.
        <br />
        <span style={{ color: "#D6FA76" }}>Your next move.</span>
      </Interactive.Div>
      <Interactive.Div
        name="Webhook transport"
        style={{
          position: "absolute",
          left: 80,
          top: 505,
          width: 920,
          height: 123,
          padding: "0 36px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "#243540",
          border: "1px solid #5D7380",
          borderRadius: 20,
          fontSize: 53,
          fontWeight: 700,
          opacity: interpolate(frame, [20, 34], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          translate: interpolate(frame, [20, 39], ["35px 0px", "0px 0px"], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          }),
        }}
      >
        Webhook<span style={{ color: "#D6FA76" }}>↗</span>
      </Interactive.Div>
      <Interactive.Div
        name="Kafka transport"
        style={{
          position: "absolute",
          left: 80,
          top: 653,
          width: 920,
          height: 123,
          padding: "0 36px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "#243540",
          border: "1px solid #5D7380",
          borderRadius: 20,
          fontSize: 53,
          fontWeight: 700,
          opacity: interpolate(frame, [49, 64], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          translate: interpolate(frame, [49, 69], ["35px 0px", "0px 0px"], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          }),
        }}
      >
        Kafka<span style={{ color: "#D6FA76" }}>↗</span>
      </Interactive.Div>
      <Interactive.Div
        name="Event Grid transport"
        style={{
          position: "absolute",
          left: 80,
          top: 801,
          width: 920,
          height: 123,
          padding: "0 36px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "#243540",
          border: "1px solid #5D7380",
          borderRadius: 20,
          fontSize: 53,
          fontWeight: 700,
          opacity: interpolate(frame, [66, 81], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          translate: interpolate(frame, [66, 86], ["35px 0px", "0px 0px"], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: Easing.out(Easing.cubic),
          }),
        }}
      >
        Azure Event Grid<span style={{ color: "#D6FA76" }}>↗</span>
      </Interactive.Div>
      <Interactive.Div
        name="Customer response options"
        style={{
          position: "absolute",
          left: 80,
          top: 981,
          fontSize: 37,
          fontWeight: 600,
          color: "#D6FA76",
          opacity: interpolate(frame, [125, 146], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Alert an operator. Escalate. Investigate.
      </Interactive.Div>
      <div
        style={{
          position: "absolute",
          left: 80,
          top: 1060,
          fontFamily: mono,
          fontSize: 25,
          color: "#AEC3CC",
        }}
      >
        LITMUS DETECTS. YOUR SYSTEM RESPONDS.
      </div>
      <Narration index={5} />
    </Stage>
  );
};
