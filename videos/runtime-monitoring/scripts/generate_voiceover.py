"""Generate a standard synthetic narrator and word timings for the explainer.

Install edge-tts==7.2.8, then run this script from any directory.
This sends only the public narration script to Microsoft's speech service.
No personal voice is cloned. The checked-in MP3s make regeneration optional.
"""

import asyncio
import json
import math
import subprocess
from pathlib import Path

import edge_tts

PROJECT = Path(__file__).resolve().parents[1]
VOICE = "en-US-AndrewNeural"
FPS = 30


async def main() -> None:
    """Write per-scene audio, Remotion Caption JSON, and measured timing data."""
    scenes = json.loads((PROJECT / "narration.json").read_text(encoding="utf-8"))
    output = PROJECT / "public" / "voiceover"
    output.mkdir(parents=True, exist_ok=True)
    probe = next((PROJECT / "node_modules" / "@remotion").glob("compositor-*/ffprobe*"))
    timeline = []
    start_frame = 0
    for scene in scenes:
        audio = output / f"{scene['id']}.mp3"
        captions = []
        speech = edge_tts.Communicate(scene["text"], VOICE, rate="+4%", boundary="WordBoundary")
        with audio.open("wb") as stream:
            async for chunk in speech.stream():
                if chunk["type"] == "audio":
                    stream.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    start_ms = chunk["offset"] / 10000
                    end_ms = start_ms + chunk["duration"] / 10000
                    captions.append(
                        {
                            "text": " " + chunk["text"],
                            "startMs": start_ms,
                            "endMs": end_ms,
                            "timestampMs": (start_ms + end_ms) / 2,
                            "confidence": None,
                        }
                    )
        probe_result = subprocess.run(
            [
                str(probe),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(audio),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(json.loads(probe_result.stdout)["format"]["duration"])
        # 12 frames before narration, 18 after. Crossfades overlap 12 frames.
        frames = math.ceil(duration * FPS) + 30
        (output / f"{scene['id']}.json").write_text(
            json.dumps(captions, indent=2) + "\n", encoding="utf-8"
        )
        timeline.append(
            {
                **scene,
                "audioSeconds": duration,
                "frames": frames,
                "startFrame": start_frame,
                "voice": VOICE,
            }
        )
        start_frame += frames - 12
        print(f"{scene['id']}: {duration:.2f}s, {frames} frames", flush=True)
    timeline_path = PROJECT / "src" / "timeline.json"
    timeline_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
    print(f"Composition: {start_frame + 12} frames, {(start_frame + 12) / FPS:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
