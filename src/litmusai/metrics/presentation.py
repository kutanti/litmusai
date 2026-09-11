"""Display task metrics consistently in CLI, Markdown and HTML reports."""

from __future__ import annotations

import html
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

HEADERS = ["Metric group", "Precision", "Recall", "F1", "Support", "TP", "FP", "FN"]


def _value(metric: dict[str, Any]) -> str:
    return f"{metric['value']:.4f}" + ("" if metric["defined"] else "*")


def _rows(metrics: dict[str, Any]) -> list[list[str]]:
    if metrics["task_type"] == "classification":
        groups = [(f"Class: {label}", values) for label, values in metrics["per_class"].items()]
        groups += [(name.title(), metrics[name]) for name in ("micro", "macro", "weighted")]
    else:
        groups = [("All items", metrics)]
    return [[name, *[_value(values[key]) for key in ("precision", "recall", "f1")],
             *[str(values.get(key, "-")) for key in ("support", "tp", "fp", "fn")]]
            for name, values in groups]


def _summary(metrics: dict[str, Any]) -> str:
    accuracy = metrics["accuracy" if metrics["task_type"] == "classification"
                       else "exact_match_accuracy"]
    label = "Accuracy" if metrics["task_type"] == "classification" else "Exact match accuracy"
    return (
        f"{label}: {_value(accuracy)} | "
        f"Prediction coverage: {metrics['valid_predictions']}/{metrics['total']} "
        f"({_value(metrics['prediction_coverage'])}) | "
        f"Invalid predictions: {metrics['invalid_predictions']} | "
        f"Execution errors: {metrics['execution_errors']}"
    )


def _matrix(metrics: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    matrix = metrics["confusion_matrix"]
    headers = ["Expected / predicted", *[
        "Invalid prediction" if label is None else label for label in matrix["predicted_labels"]
    ]]
    rows = [[label, *map(str, counts)] for label, counts in
            zip(matrix["expected_labels"], matrix["counts"], strict=True)]
    return headers, rows


def print_metrics(metrics: dict[str, Any], console: Console) -> None:
    """Print task counts, averages, coverage, and the classification confusion matrix."""
    table = Table(title=f"{metrics['task_type'].title()} metrics")
    for name in HEADERS:
        table.add_column(name)
    for row in _rows(metrics):
        # Text disables Rich markup in user-supplied class labels.
        table.add_row(*(Text(cell) for cell in row))
    console.print(table)
    console.print(_summary(metrics), markup=False)
    console.print("* Zero-denominator values contribute 0; their metric or average is undefined.")
    console.print("Assertion scores and pass rates are reported separately.")
    if metrics["task_type"] == "classification":
        headers, rows = _matrix(metrics)
        matrix = Table(title="Confusion matrix")
        for name in headers:
            matrix.add_column(Text(name))
        for row in rows:
            matrix.add_row(*(Text(cell) for cell in row))
        console.print(matrix)


def metric_markdown(metrics: dict[str, Any]) -> str:
    """Render labeled scores as Markdown with escaped labels."""
    def table(headers: list[str], rows: list[list[str]]) -> str:
        def line(cells: list[str]) -> str:
            return "| " + " | ".join(html.escape(c).replace("|", "&#124;").replace("\n", " ")
                                      for c in cells) + " |"
        return "\n".join([line(headers), line(["---"] * len(headers)), *map(line, rows)])

    parts = [f"### {metrics['task_type'].title()} metrics", _summary(metrics),
             table(HEADERS, _rows(metrics)),
             "* marks metrics or averages with undefined denominators; zero values are used.",
             "Assertion scores and pass rates are reported separately."]
    if metrics["task_type"] == "classification":
        headers, rows = _matrix(metrics)
        parts.extend(["Confusion matrix:", table(headers, rows)])
    return "\n\n".join(parts)


def metric_html(metrics: dict[str, Any]) -> str:
    """Render escaped, self-contained HTML tables for labeled metrics."""
    def table(headers: list[str], rows: list[list[str]]) -> str:
        head = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                       for row in rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    parts = [f"<h2>{html.escape(metrics['task_type'].title())} metrics</h2>",
             f"<p>{html.escape(_summary(metrics))}</p>", table(HEADERS, _rows(metrics)),
             "<p>* Zero-denominator values contribute 0 and are flagged undefined.</p>",
             "<p>Assertion scores and pass rates are reported separately.</p>"]
    if metrics["task_type"] == "classification":
        headers, rows = _matrix(metrics)
        parts.extend(["<h3>Confusion matrix</h3>", table(headers, rows)])
    return '<section class="section">' + "\n".join(parts) + "</section>"
