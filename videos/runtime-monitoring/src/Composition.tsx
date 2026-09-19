import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { Tests } from "./scenes/Tests";
import { Support } from "./scenes/Support";
import { Attack } from "./scenes/Attack";
import { Monitor } from "./scenes/Monitor";
import { Detect } from "./scenes/Detect";
import { Route } from "./scenes/Route";
import { Policies } from "./scenes/Policies";
import { Close } from "./scenes/Close";

export const RuntimeStory = () => (
  <TransitionSeries>
    <TransitionSeries.Sequence durationInFrames={205} name="Tests pass">
      <Tests />
    </TransitionSeries.Sequence>
    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 12 })}
    />
    <TransitionSeries.Sequence
      durationInFrames={177}
      name="Normal support conversation"
    >
      <Support />
    </TransitionSeries.Sequence>
    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 12 })}
    />
    <TransitionSeries.Sequence
      durationInFrames={244}
      name="The malicious request"
    >
      <Attack />
    </TransitionSeries.Sequence>
    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 12 })}
    />
    <TransitionSeries.Sequence
      durationInFrames={206}
      name="Asynchronous capture"
    >
      <Monitor />
    </TransitionSeries.Sequence>
    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 12 })}
    />
    <TransitionSeries.Sequence
      durationInFrames={235}
      name="Evidence-bearing event"
    >
      <Detect />
    </TransitionSeries.Sequence>
    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 12 })}
    />
    <TransitionSeries.Sequence durationInFrames={203} name="Customer response">
      <Route />
    </TransitionSeries.Sequence>
    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 12 })}
    />
    <TransitionSeries.Sequence
      durationInFrames={234}
      name="More configurable policies"
    >
      <Policies />
    </TransitionSeries.Sequence>
    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 12 })}
    />
    <TransitionSeries.Sequence durationInFrames={204} name="Test and monitor">
      <Close />
    </TransitionSeries.Sequence>
  </TransitionSeries>
);
