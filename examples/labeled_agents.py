"""Local agents for the routing and extraction metric examples; no API keys needed."""

import json
import re


def route(task: str) -> str:
    """Route invoice and payment requests to billing; everything else to support."""
    label = "billing" if any(word in task.lower() for word in ("invoice", "payment")) else "support"
    return json.dumps({"label": label})


def extract(task: str) -> str:
    """Return email occurrences, including duplicates, for extraction scoring."""
    emails = re.findall(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", task)
    return json.dumps({"emails": emails})
