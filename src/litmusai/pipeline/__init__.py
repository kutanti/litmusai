"""Pipeline — chain eval + safety + report in one call.

Usage::

    from litmusai.pipeline import Pipeline

    pipeline = Pipeline(
        agent=my_agent,
        suite="coding",           # built-in suite name or TestSuite object
        safety=True,              # run safety scan after eval
        runs=3,                   # multi-run for statistical confidence
        report="html",            # generate HTML report
        log_dir="./eval-logs",    # persist results
    )

    result = await pipeline.run()

    print(result.eval.summary())        # eval results
    print(result.safety.safety_score)   # safety score
    print(result.report_path)           # path to HTML report
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litmusai.core.agent import Agent
from litmusai.core.runner import EvalResults, MultiRunResults
from litmusai.core.scorer import Scorer
from litmusai.core.suite import TestSuite


@dataclass
class PipelineResult:
    """Result from a full pipeline run.

    Attributes:
        eval: Evaluation results (single run or first run if multi-run).
        multi_run: Multi-run statistics (``None`` if ``runs=1``).
        safety: Safety scan report (``None`` if safety not enabled).
        report_path: Path to generated report (``None`` if no report).
        baseline_diff: Baseline comparison (``None`` if no baseline).
        duration_ms: Total pipeline wall-clock time in milliseconds.
    """

    eval: EvalResults
    multi_run: MultiRunResults | None = None
    safety: Any = None  # SafetyReport — lazy import to avoid circular
    report_path: str | None = None
    baseline_diff: Any = None  # DiffSummary
    duration_ms: float = 0.0
    threshold: float = 0.5

    def summary(self) -> str:
        """One-line summary of the full pipeline run."""
        parts = [self.eval.summary()]

        if self.safety is not None:
            score = self.safety.safety_score
            verdict = "SAFE" if self.safety.is_safe else "UNSAFE"
            parts.append(f"🛡️ {score:.0f}/100 {verdict}")

        if self.multi_run is not None:
            flaky = len(self.multi_run.flaky_tests)
            if flaky:
                parts.append(f"⚠️ {flaky} flaky tests")
            else:
                parts.append(f"📊 {self.multi_run.n_runs} runs — stable")

        if self.report_path:
            parts.append(f"📄 {self.report_path}")

        parts.append(f"⏱️ {self.duration_ms:.0f}ms total")
        return " | ".join(parts)

    @property
    def passed(self) -> bool:
        """True if eval passed threshold AND safety passed (if enabled)."""
        eval_ok = self.eval.pass_rate >= self.threshold
        safety_ok = self.safety.is_safe if self.safety else True
        return eval_ok and safety_ok


class Pipeline:
    """Chain eval + safety + report in one call.

    Orchestrates the full evaluation workflow:
    ``evaluate()`` / ``multi_evaluate()`` → ``SafetyScanner`` →
    baseline ``diff_results()`` → report generation.

    Cost tracking is handled automatically by the underlying
    ``evaluate()`` call when model pricing is registered.
    LLM-as-judge scoring is available via assertions
    (``LLMGrade``) on individual test cases.

    Args:
        agent: The agent to evaluate.
        suite: Suite name (string for built-in) or :class:`TestSuite` object.
        scorer: Custom scorer (optional).
        concurrency: Max parallel evaluations.
        safety: Run safety scan after evaluation.
        safety_depth: Safety scan depth — ``"basic"``, ``"standard"``,
            or ``"thorough"``.
        runs: Number of evaluation runs (>1 enables multi-run stats).
        report: Report format — ``"html"``, ``"junit"``, ``"csv"``,
            or ``None``.
        report_path: Output path for the report. Auto-generated if not set.
        log_dir: Directory to persist evaluation results as JSON.
        baseline: Path to a previous result JSON for regression detection.
        threshold: Minimum pass rate (0.0–1.0) for the pipeline to "pass".
        verbose: Show progress output.

    Example::

        pipeline = Pipeline(
            agent=my_agent,
            suite="coding",
            safety=True,
            runs=3,
            report="html",
        )
        result = await pipeline.run()
    """

    def __init__(
        self,
        agent: Agent,
        suite: TestSuite | str,
        *,
        scorer: Scorer | None = None,
        concurrency: int = 5,
        safety: bool = False,
        safety_depth: str = "standard",
        runs: int = 1,
        report: str | None = None,
        report_path: str | None = None,
        log_dir: str | Path | None = None,
        baseline: str | Path | None = None,
        threshold: float = 0.5,
        verbose: bool = True,
    ):
        self.agent = agent
        self.scorer = scorer
        self.concurrency = max(1, concurrency)
        self.safety = safety
        self.safety_depth = safety_depth
        self.runs = max(1, runs)
        self.report = report
        self.report_path = report_path
        self.log_dir = log_dir
        self.baseline = baseline
        self.threshold = threshold
        self.verbose = verbose

        # Resolve suite
        if isinstance(suite, str):
            self._suite = TestSuite.load(suite)
        else:
            self._suite = suite

    async def run(self) -> PipelineResult:
        """Execute the full pipeline.

        Steps:
            1. Evaluate agent against suite (single or multi-run)
            2. Run safety scan (if enabled)
            3. Compare with baseline (if provided)
            4. Generate report (if requested)

        Returns:
            :class:`PipelineResult` with all outputs.
        """
        start = time.perf_counter()

        # ── Step 1: Evaluate ─────────────────────────────────────
        eval_results: EvalResults
        multi_results: MultiRunResults | None = None

        if self.runs > 1:
            from litmusai.core.runner import multi_evaluate

            if self.log_dir:
                import warnings
                warnings.warn(
                    "log_dir is not supported with multi-run (runs > 1). "
                    "Logs will not be saved.",
                    stacklevel=2,
                )

            multi_results = await multi_evaluate(
                self.agent,
                self._suite,
                runs=self.runs,
                scorer=self.scorer,
                concurrency=self.concurrency,
                verbose=self.verbose,
            )
            # Use the first run as the primary eval result
            eval_results = multi_results.run_results[0]
        else:
            from litmusai.core.runner import evaluate

            eval_results = await evaluate(
                self.agent,
                self._suite,
                scorer=self.scorer,
                concurrency=self.concurrency,
                verbose=self.verbose,
                log_dir=str(self.log_dir) if self.log_dir else None,
            )

        # ── Step 2: Safety scan ──────────────────────────────────
        safety_report = None
        if self.safety:
            from litmusai.safety import SafetyScanner

            scanner = SafetyScanner(depth=self.safety_depth)
            safety_report = await scanner.scan(self.agent)

        # ── Step 3: Baseline comparison ──────────────────────────
        baseline_diff = None
        if self.baseline:
            from litmusai.results import diff_results, load_results

            baseline_data = load_results(str(self.baseline))
            current_data = eval_results.to_dict()
            baseline_diff = diff_results(baseline_data, current_data)

        # ── Step 4: Generate report ──────────────────────────────
        generated_report_path: str | None = None
        if self.report:
            generated_report_path = self._generate_report(
                eval_results, safety_report,
            )

        duration = (time.perf_counter() - start) * 1000

        result = PipelineResult(
            eval=eval_results,
            multi_run=multi_results,
            safety=safety_report,
            report_path=generated_report_path,
            baseline_diff=baseline_diff,
            duration_ms=duration,
            threshold=self.threshold,
        )

        if self.verbose:
            from rich.console import Console
            Console().print(f"\n[bold]{result.summary()}[/bold]\n")

        return result

    def _generate_report(
        self,
        eval_results: EvalResults,
        safety_report: Any,
    ) -> str:
        """Generate report in the requested format."""
        fmt = (self.report or "").lower()
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if fmt == "html":
            from litmusai.reports import render_html

            path = self.report_path or f"report_{timestamp}.html"
            render_html(eval_results.to_dict(), path)
            return path

        if fmt == "junit":
            from litmusai.exports import to_junit_xml

            path = self.report_path or f"results_{timestamp}.xml"
            to_junit_xml(eval_results.to_dict(), path)
            return path

        if fmt == "csv":
            from litmusai.exports import to_csv

            path = self.report_path or f"results_{timestamp}.csv"
            to_csv(eval_results.to_dict(), path)
            return path

        msg = f"Unknown report format: {fmt!r}. Use 'html', 'junit', or 'csv'."
        raise ValueError(msg)


# ─── Convenience function ────────────────────────────────────────


async def run_pipeline(
    agent: Agent,
    suite: TestSuite | str,
    **kwargs: Any,
) -> PipelineResult:
    """Convenience function to create and run a pipeline.

    Same args as :class:`Pipeline`. Returns :class:`PipelineResult`.

    Example::

        result = await run_pipeline(
            agent, "coding", safety=True, report="html",
        )
    """
    return await Pipeline(agent, suite, **kwargs).run()
