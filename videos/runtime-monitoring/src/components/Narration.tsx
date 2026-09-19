import { Audio } from "@remotion/media";
import { Sequence, staticFile } from "remotion";
import { Captions } from "./Captions";
import tests from "../../public/voiceover/01-tests.json";
import support from "../../public/voiceover/02-support.json";
import attack from "../../public/voiceover/03-attack.json";
import monitor from "../../public/voiceover/04-monitor.json";
import detect from "../../public/voiceover/05-detect.json";
import route from "../../public/voiceover/06-route.json";
import policies from "../../public/voiceover/07-policies.json";
import close from "../../public/voiceover/08-close.json";

const captions = [
  tests,
  support,
  attack,
  monitor,
  detect,
  route,
  policies,
  close,
];
const names = [
  "01-tests",
  "02-support",
  "03-attack",
  "04-monitor",
  "05-detect",
  "06-route",
  "07-policies",
  "08-close",
];

export const Narration: React.FC<{ index: number; light?: boolean }> = ({
  index,
  light,
}) => (
  <Sequence from={12} name="Narration and captions" layout="none">
    <Audio src={staticFile("voiceover/" + names[index] + ".mp3")} volume={1} />
    <Captions captions={captions[index]} light={light} />
  </Sequence>
);
