"""Tests for HTML report generation."""

import json

from litmusai.reports import render_html


def _make_data(
    agent: str = "test-agent",
    suite: str = "test-suite",
    n_pass: int = 3,
    n_fail: int = 1,
) -> dict:
    results = []
    for i in range(n_pass):
        results.append({
            "case_id": f"pass_{i}",
            "case_name": f"Passing test {i}",
            "task": f"Task {i}",
            "response": f"Response {i}",
            "passed": True,
            "score": 1.0,
            "score_reason": "All assertions passed",
            "latency_ms": 100 + i * 50,
            "cost": 0.001,
            "input_tokens": 20,
            "output_tokens": 10,
        })
    for i in range(n_fail):
        results.append({
            "case_id": f"fail_{i}",
            "case_name": f"Failing test {i}",
            "task": f"Task fail {i}",
            "response": f"Wrong response {i}",
            "passed": False,
            "score": 0.3,
            "score_reason": "Assertion failed",
            "latency_ms": 200,
            "cost": 0.002,
            "input_tokens": 25,
            "output_tokens": 15,
        })

    total = n_pass + n_fail
    return {
        "agent_name": agent,
        "suite_name": suite,
        "timestamp": "2026-04-03T10:00:00",
        "summary": {
            "total": total,
            "passed": n_pass,
            "failed": n_fail,
            "pass_rate": n_pass / total if total else 0,
            "avg_score": (
                (n_pass * 1.0 + n_fail * 0.3) / total
                if total else 0
            ),
            "avg_latency_ms": 150.0,
            "total_cost": n_pass * 0.001 + n_fail * 0.002,
            "total_input_tokens": n_pass * 20 + n_fail * 25,
            "total_output_tokens": n_pass * 10 + n_fail * 15,
        },
        "results": results,
    }


class TestRenderHtml:
    def test_basic_render(self, tmp_path):
        data = _make_data()
        path = render_html(data, tmp_path / "report.html")
        assert path.exists()
        html = path.read_text()
        assert "LitmusAI Report" in html
        assert "test-agent" in html
        assert "test-suite" in html

    def test_contains_all_tests(self, tmp_path):
        data = _make_data(n_pass=3, n_fail=2)
        path = render_html(data, tmp_path / "report.html")
        html = path.read_text()
        assert "Passing test 0" in html
        assert "Passing test 1" in html
        assert "Passing test 2" in html
        assert "Failing test 0" in html
        assert "Failing test 1" in html

    def test_pass_fail_status(self, tmp_path):
        data = _make_data(n_pass=1, n_fail=1)
        path = render_html(data, tmp_path / "report.html")
        html = path.read_text()
        assert "PASS" in html
        assert "FAIL" in html

    def test_summary_cards(self, tmp_path):
        data = _make_data()
        path = render_html(data, tmp_path / "report.html")
        html = path.read_text()
        assert "Pass Rate" in html
        assert "Total Cost" in html
        assert "Avg Latency" in html
        assert "Tokens" in html

    def test_filter_buttons(self, tmp_path):
        data = _make_data()
        path = render_html(data, tmp_path / "report.html")
        html = path.read_text()
        assert "filterTests" in html
        assert "Passed" in html
        assert "Failed" in html

    def test_sort_script(self, tmp_path):
        data = _make_data()
        path = render_html(data, tmp_path / "report.html")
        html = path.read_text()
        assert "sortTable" in html

    def test_creates_parent_dirs(self, tmp_path):
        data = _make_data()
        path = render_html(
            data, tmp_path / "deep" / "nested" / "report.html",
        )
        assert path.exists()

    def test_empty_results(self, tmp_path):
        data = _make_data(n_pass=0, n_fail=0)
        path = render_html(data, tmp_path / "report.html")
        assert path.exists()
        html = path.read_text()
        assert "0%" in html

    def test_all_pass(self, tmp_path):
        data = _make_data(n_pass=5, n_fail=0)
        path = render_html(data, tmp_path / "report.html")
        html = path.read_text()
        assert "100%" in html

    def test_self_contained(self, tmp_path):
        """HTML should be fully self-contained — no external CSS/JS."""
        data = _make_data()
        path = render_html(data, tmp_path / "report.html")
        html = path.read_text()
        assert "<style>" in html
        assert "<script>" in html
        # No external links
        assert "stylesheet" not in html.lower()
        assert 'src="http' not in html

    def test_dark_theme(self, tmp_path):
        data = _make_data()
        path = render_html(data, tmp_path / "report.html")
        html = path.read_text()
        assert "#0d1117" in html  # dark background

    def test_detail_rows(self, tmp_path):
        data = _make_data()
        path = render_html(data, tmp_path / "report.html")
        html = path.read_text()
        assert "detail-row" in html
        assert "toggleDetail" in html

    def test_valid_html(self, tmp_path):
        data = _make_data()
        path = render_html(data, tmp_path / "report.html")
        html = path.read_text()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_score_bar(self, tmp_path):
        data = _make_data()
        path = render_html(data, tmp_path / "report.html")
        html = path.read_text()
        assert "score-bar" in html

    def test_litmusai_footer(self, tmp_path):
        data = _make_data()
        path = render_html(data, tmp_path / "report.html")
        html = path.read_text()
        assert "kutanti/litmusai" in html

    def test_xss_protection(self, tmp_path):
        """User-controlled text must be HTML-escaped."""
        data = _make_data(agent='<script>alert("xss")</script>')
        data["results"][0]["case_name"] = '<img onerror="alert(1)">'
        data["results"][0]["response"] = '"><script>evil</script>'
        path = render_html(data, tmp_path / "report.html")
        html_content = path.read_text()
        # Raw tags must NOT appear
        assert "<script>alert" not in html_content
        assert '<img onerror' not in html_content
        # Escaped versions should appear
        assert "&lt;script&gt;" in html_content

    def test_safe_dom_ids(self, tmp_path):
        """Case IDs with special chars must produce safe DOM IDs."""
        data = _make_data(n_pass=1, n_fail=0)
        data["results"][0]["case_id"] = "test case/with spaces&quotes"
        path = render_html(data, tmp_path / "report.html")
        html_content = path.read_text()
        # Should not contain raw special chars in IDs
        assert 'id="detail-test_case' in html_content

    def test_sort_data_values(self, tmp_path):
        """Cost and latency cells should have data-value for sorting."""
        data = _make_data()
        path = render_html(data, tmp_path / "report.html")
        html_content = path.read_text()
        assert "data-value=" in html_content


class TestCliHtml:
    def test_report_html_flag(self):
        """CLI report command has --html flag."""
        from click.testing import CliRunner

        from litmusai.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--help"])
        assert "--html" in result.output

    def test_report_html_generation(self, tmp_path):
        from click.testing import CliRunner

        from litmusai.cli.main import cli

        runner = CliRunner()
        # Create a results JSON
        data = _make_data()
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(data))
        html_file = tmp_path / "report.html"

        result = runner.invoke(
            cli,
            ["report", "-r", str(results_file),
             "--html", str(html_file)],
        )
        assert result.exit_code == 0
        assert html_file.exists()
        assert "HTML report saved" in result.output
