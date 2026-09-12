"""JUnit XML export for CI integration.

Generates JUnit XML reports from evaluation results, compatible
with GitHub Actions, Jenkins, GitLab CI, and other CI systems.

Usage::

    from litmusai.exports import to_junit_xml
    results = await evaluate(agent, suite)
    to_junit_xml(results.to_dict(), "results.xml")
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from litmusai.results import normalize_results


def to_junit_xml(
    data: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Export evaluation results as JUnit XML.

    Args:
        data: Result dict from ``EvalResults.to_dict()`` or a CLI status envelope.
        output_path: Where to write the XML file.

    Returns:
        Path to the generated XML file.
    """
    data = normalize_results(data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = data.get("summary", {})
    results = data.get("results", [])
    agent_name = data.get("agent_name", "agent")
    suite_name = data.get("suite_name", "suite")

    total = summary.get("total", len(results))
    failures = summary.get("failed", sum(
        1 for r in results if not r.get("passed")
    ))
    time_s = summary.get("avg_latency_ms", 0) * total / 1000

    # Build XML
    testsuites = ET.Element("testsuites")
    testsuite = ET.SubElement(testsuites, "testsuite")
    testsuite.set("name", f"{agent_name}/{suite_name}")
    testsuite.set("tests", str(total))
    testsuite.set("failures", str(failures))
    testsuite.set("errors", "0")
    testsuite.set("time", f"{time_s:.3f}")
    testsuite.set("timestamp", data.get("timestamp", ""))

    # Properties
    props = ET.SubElement(testsuite, "properties")
    for key, value in [
        ("agent", agent_name),
        ("suite", suite_name),
        ("pass_rate", str(summary.get("pass_rate", 0))),
        ("total_cost", str(summary.get("total_cost", 0))),
        ("avg_latency_ms", str(summary.get("avg_latency_ms", 0))),
    ]:
        prop = ET.SubElement(props, "property")
        prop.set("name", key)
        prop.set("value", value)
    if data.get("dataset"):
        ET.SubElement(props, "property", name="dataset", value=json.dumps(
            data["dataset"], ensure_ascii=False))

    # Test cases
    for r in results:
        tc = ET.SubElement(testsuite, "testcase")
        name = r.get("case_name", r.get("case_id", "?"))
        identity = []
        if r.get("case_id") is not None:
            identity.append(f"case_id={r['case_id']}")
        if r.get("repetition") is not None:
            identity.append(f"run {r['repetition']}")
        if identity:
            name = f"{name} [{'; '.join(identity)}]"
        tc.set("name", name)
        tc.set("classname", f"{agent_name}.{suite_name}")
        tc.set("time", f"{r.get('latency_ms', 0) / 1000:.3f}")

        identities = {key: r[key] for key in ("evaluation_id", "case_id", "repetition")
                      if r.get(key) is not None}
        for key in ("inputs", "metadata", "source", "ground_truth", "response_metadata"):
            if r.get(key) is not None:
                identities[key] = json.dumps(r[key], ensure_ascii=False)
        if identities:
            case_props = ET.SubElement(tc, "properties")
            for key, value in identities.items():
                ET.SubElement(case_props, "property", name=key, value=str(value))

        if not r.get("passed", True):
            failure = ET.SubElement(tc, "failure")
            failure.set("message", r.get("score_reason", "Failed"))
            failure.set("type", "AssertionError")
            # Include response in failure body
            response = str(r.get("response", ""))[:2000]
            failure.text = (
                f"Task: {r.get('task', '')}\n"
                f"Score: {r.get('score', 0)}\n"
                f"Response: {response}"
            )

        # System output — include response for all cases
        stdout = ET.SubElement(tc, "system-out")
        stdout.text = str(r.get("response", ""))

    # Write with proper XML declaration
    tree = ET.ElementTree(testsuites)
    ET.indent(tree, space="  ")
    tree.write(
        str(output_path),
        encoding="UTF-8",
        xml_declaration=True,
    )

    return output_path


def to_csv(
    data: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Export evaluation results as CSV.

    Args:
        data: Result dict from ``EvalResults.to_dict()`` or a CLI status envelope.
        output_path: Where to write the CSV file.

    Returns:
        Path to the generated CSV file.
    """
    import csv

    data = normalize_results(data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = data.get("results", [])

    fieldnames = [
        "case_id", "case_name", "task", "passed", "score",
        "score_reason", "latency_ms", "cost",
        "input_tokens", "output_tokens", "model",
        "response", "evaluation_id", "repetition",
        "dataset_id", "dataset_revision", "dataset_fingerprint", "dataset_metadata",
        "dataset_source",
        "inputs", "metadata", "source", "ground_truth", "response_metadata",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            dataset = data.get("dataset") or {}
            for key in ("id", "revision", "fingerprint"):
                row[f"dataset_{key}"] = dataset.get(key) or ""
            row["dataset_metadata"] = (
                json.dumps(dataset.get("metadata", {}), ensure_ascii=False) if dataset else ""
            )
            row["dataset_source"] = (
                json.dumps(dataset["source"], ensure_ascii=False)
                if dataset.get("source") is not None else ""
            )
            for key in ("inputs", "metadata", "source", "ground_truth", "response_metadata"):
                row[key] = json.dumps(r[key], ensure_ascii=False) if r.get(key) is not None else ""
            writer.writerow(row)

    return output_path
