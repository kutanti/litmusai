import "./index.css";
import { Composition } from "remotion";
import { RuntimeStory } from "./Composition";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="RuntimeStory"
        component={RuntimeStory}
        durationInFrames={1624}
        fps={30}
        width={1080}
        height={1350}
      />
    </>
  );
};
