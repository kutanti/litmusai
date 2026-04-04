"""HTML report generator for LitmusAI evaluation results.

Renders results as a self-contained HTML file with summary
cards, tables, and filtering — no external dependencies.

Usage::

    from litmusai.reports import render_html
    results = await evaluate(agent, suite)
    render_html(results.to_dict(), "report.html")
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any


def _esc(text: str) -> str:
    """HTML-escape text content."""
    return html.escape(str(text), quote=True)


def _safe_id(raw: str) -> str:
    """Derive a safe DOM id from a case id."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(raw))


_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LitmusAI Report — {title}</title>
<style>
:root {{
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
    --blue: #58a6ff;
    --purple: #bc8cff;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
    max-width: 1200px;
    margin: 0 auto;
}}
h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
h2 {{ font-size: 1.3rem; margin: 2rem 0 1rem; color: var(--blue); }}
.subtitle {{ color: var(--text-dim); margin-bottom: 2rem; }}
.cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}}
.card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem;
}}
.card .label {{ color: var(--text-dim); font-size: 0.85rem; }}
.card .value {{ font-size: 1.8rem; font-weight: 700; margin-top: 0.3rem; }}
.card .value.green {{ color: var(--green); }}
.card .value.red {{ color: var(--red); }}
.card .value.yellow {{ color: var(--yellow); }}
.pass-bar {{
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    margin-top: 0.5rem;
    overflow: hidden;
}}
.pass-bar .fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    border-radius: 8px;
    overflow: hidden;
}}
th, td {{
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}}
th {{
    background: rgba(0,0,0,0.3);
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    color: var(--text-dim);
    cursor: pointer;
    user-select: none;
}}
th:hover {{ color: var(--blue); }}
tr:last-child td {{ border-bottom: none; }}
tr:hover {{ background: rgba(255,255,255,0.03); }}
.pass {{ color: var(--green); font-weight: 600; }}
.fail {{ color: var(--red); font-weight: 600; }}
.score-bar {{
    display: inline-block;
    width: 60px;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    margin-right: 8px;
    vertical-align: middle;
}}
.score-bar .fill {{
    height: 100%;
    border-radius: 3px;
}}
.filter-bar {{
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}}
.filter-btn {{
    padding: 0.4rem 0.8rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-dim);
    cursor: pointer;
    font-size: 0.85rem;
    transition: all 0.2s;
}}
.filter-btn:hover, .filter-btn.active {{
    border-color: var(--blue);
    color: var(--blue);
}}
.detail-row {{ display: none; }}
.detail-row td {{
    padding: 0.5rem 1rem 1rem 3rem;
    color: var(--text-dim);
    font-size: 0.9rem;
}}
.detail-row.open {{ display: table-row; }}
.expand {{ cursor: pointer; color: var(--text-dim); }}
.expand:hover {{ color: var(--blue); }}
footer {{
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    color: var(--text-dim);
    font-size: 0.8rem;
    text-align: center;
}}
footer a {{ color: var(--blue); text-decoration: none; }}
</style>
</head>
<body>
<h1>🧪 {title}</h1>
<p class="subtitle">{subtitle}</p>

<div class="cards">
    <div class="card">
        <div class="label">Pass Rate</div>
        <div class="value {pass_color}">{pass_rate}</div>
        <div class="pass-bar">
            <div class="fill" style="width:{pass_pct}%;background:var(--{pass_color})"></div>
        </div>
    </div>
    <div class="card">
        <div class="label">Tests</div>
        <div class="value">{total}</div>
        <div style="color:var(--text-dim);font-size:0.85rem">
            ✅ {passed} passed · ❌ {failed} failed
        </div>
    </div>
    <div class="card">
        <div class="label">Avg Score</div>
        <div class="value">{avg_score}</div>
    </div>
    <div class="card">
        <div class="label">Total Cost</div>
        <div class="value">${total_cost}</div>
    </div>
    <div class="card">
        <div class="label">Avg Latency</div>
        <div class="value">{avg_latency}</div>
    </div>
    <div class="card">
        <div class="label">Tokens</div>
        <div class="value">{total_tokens}</div>
        <div style="color:var(--text-dim);font-size:0.85rem">
            ↑{input_tokens} · ↓{output_tokens}
        </div>
    </div>
</div>

<h2>Test Results</h2>

<div class="filter-bar">
    <button class="filter-btn active" onclick="filterTests('all', this)">All ({total})</button>
    <button class="filter-btn" onclick="filterTests('pass', this)">✅ Passed ({passed})</button>
    <button class="filter-btn" onclick="filterTests('fail', this)">❌ Failed ({failed})</button>
</div>

<table id="results-table">
<thead>
<tr>
    <th style="width:30px"></th>
    <th onclick="sortTable(1)">Test</th>
    <th onclick="sortTable(2)" style="width:80px">Status</th>
    <th onclick="sortTable(3)" style="width:100px">Score</th>
    <th onclick="sortTable(4)" style="width:100px">Latency</th>
    <th onclick="sortTable(5)" style="width:80px">Cost</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<footer>
    Generated by <a href="https://github.com/kutanti/litmusai">LitmusAI</a> · {timestamp}
</footer>

<script>
function filterTests(type, btn) {{
    document.querySelectorAll('.filter-btn').forEach(function(b) {{
        b.classList.remove('active');
    }});
    if (btn) btn.classList.add('active');
    document.querySelectorAll('#results-table tbody tr.result-row').forEach(function(row) {{
        var status = row.dataset.status;
        row.style.display = (type === 'all' || type === status) ? '' : 'none';
        var detail = row.nextElementSibling;
        if (detail && detail.classList.contains('detail-row')) {{
            detail.classList.remove('open');
        }}
    }});
}}

function toggleDetail(safeId) {{
    var el = document.getElementById('detail-' + safeId);
    if (el) el.classList.toggle('open');
}}

function sortTable(col) {{
    var table = document.getElementById('results-table');
    var tbody = table.querySelector('tbody');
    var rows = Array.prototype.slice.call(
        tbody.querySelectorAll('tr.result-row')
    );
    var details = {{}};
    rows.forEach(function(r) {{
        var next = r.nextElementSibling;
        if (next && next.classList.contains('detail-row')) {{
            details[r.dataset.safeid] = next;
        }}
    }});
    var dir = table.dataset.sortDir === 'asc' ? 'desc' : 'asc';
    table.dataset.sortDir = dir;
    rows.sort(function(a, b) {{
        var va = a.cells[col].dataset.value || a.cells[col].textContent.trim();
        var vb = b.cells[col].dataset.value || b.cells[col].textContent.trim();
        var na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) return dir === 'asc' ? na - nb : nb - na;
        return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
    }});
    tbody.innerHTML = '';
    rows.forEach(function(r) {{
        tbody.appendChild(r);
        if (details[r.dataset.safeid]) tbody.appendChild(details[r.dataset.safeid]);
    }});
}}
</script>
</body>
</html>"""


def _score_color(score: float) -> str:
    if score >= 0.9:
        return "green"
    if score >= 0.7:
        return "yellow"
    return "red"


def _make_row(idx: int, r: dict[str, Any]) -> str:
    """Generate HTML for one test result row + detail row."""
    passed = r.get("passed", False)
    status_cls = "pass" if passed else "fail"
    status_txt = "PASS" if passed else "FAIL"
    score = r.get("score", 0)
    score_pct = score * 100
    color = _score_color(score)
    latency = r.get("latency_ms", 0)
    cost = r.get("cost", 0)
    case_id = r.get("case_id", f"case_{idx}")
    safe_id = _safe_id(case_id)
    case_name = _esc(r.get("case_name", case_id))
    reason = _esc(r.get("score_reason", ""))
    response = _esc(str(r.get("response", ""))[:500])
    task = _esc(str(r.get("task", ""))[:200])

    row = (
        f'<tr class="result-row" data-status="{"pass" if passed else "fail"}" '
        f'data-safeid="{safe_id}">'
        f'<td class="expand" onclick="toggleDetail(\'{safe_id}\')">▸</td>'
        f"<td>{case_name}</td>"
        f'<td><span class="{status_cls}">{status_txt}</span></td>'
        f'<td data-value="{score:.4f}">'
        f'<span class="score-bar"><span class="fill" '
        f'style="width:{score_pct}%;background:var(--{color})"></span></span>'
        f"{score:.2f}</td>"
        f'<td data-value="{latency:.1f}">{latency:.0f}ms</td>'
        f'<td data-value="{cost:.6f}">${cost:.4f}</td>'
        f"</tr>"
    )

    detail = (
        f'<tr class="detail-row" id="detail-{safe_id}">'
        f"<td colspan=\"6\">"
        f"<strong>Task:</strong> {task}<br>"
        f"<strong>Reason:</strong> {reason}<br>"
        f"<strong>Response:</strong> {response}"
        f"</td></tr>"
    )

    return row + "\n" + detail


def render_html(
    data: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Render evaluation results as a self-contained HTML report.

    Args:
        data: Result dict from ``EvalResults.to_dict()``.
        output_path: Where to write the HTML file.

    Returns:
        Path to the generated HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = data.get("summary", {})
    results = data.get("results", [])

    total = summary.get("total", len(results))
    passed = summary.get("passed", sum(
        1 for r in results if r.get("passed")
    ))
    failed = total - passed
    pass_rate_val = passed / total if total > 0 else 0
    pass_pct = pass_rate_val * 100

    rows_html = "\n".join(
        _make_row(i, r) for i, r in enumerate(results)
    )

    total_in = summary.get("total_input_tokens", 0)
    total_out = summary.get("total_output_tokens", 0)

    agent_name = _esc(data.get("agent_name", "Agent"))
    suite_name = _esc(data.get("suite_name", "Suite"))

    html_out = _TEMPLATE.format(
        title=f"{agent_name} — {suite_name}",
        subtitle=f"Agent: {agent_name} · "
                 f"Suite: {suite_name} · "
                 f"{total} tests",
        pass_rate=f"{pass_rate_val:.0%}",
        pass_pct=f"{pass_pct:.0f}",
        pass_color=_score_color(pass_rate_val),
        total=total,
        passed=passed,
        failed=failed,
        avg_score=f"{summary.get('avg_score', 0):.2f}",
        total_cost=f"{summary.get('total_cost', 0):.4f}",
        avg_latency=f"{summary.get('avg_latency_ms', 0):.0f}ms",
        total_tokens=f"{total_in + total_out:,}",
        input_tokens=f"{total_in:,}",
        output_tokens=f"{total_out:,}",
        rows=rows_html,
        timestamp=_esc(data.get("timestamp", "")),
    )

    output_path.write_text(html_out)
    return output_path
