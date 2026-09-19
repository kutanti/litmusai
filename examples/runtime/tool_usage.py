"""Run with the example collector/receiver to emit one conversation usage alert."""

import asyncio
import os

from examples.runtime.agent import send_email
from litmusai.runtime import RuntimeClient


async def main() -> None:
    """Make 21 simulated, authorized calls; the configured limit is 20 in 60 seconds."""
    async with RuntimeClient(
        endpoint="http://127.0.0.1:8765",
        api_key=os.environ["LITMUS_RUNTIME_API_KEY"],
        project_id="support",
        allow_local_http=True,
    ) as monitor:
        async with monitor.session(agent_id="support-agent") as session:
            monitored_send = session.wrap_tool(send_email, destination_argument="recipient")
            for index in range(21):
                await monitored_send("support@example.com", f"Simulated request {index + 1}")
            await monitor.aflush()
            print(monitor.health)


if __name__ == "__main__":
    asyncio.run(main())
