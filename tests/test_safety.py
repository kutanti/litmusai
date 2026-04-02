"""Tests for the Agent Safety & Red Teaming Suite."""

from __future__ import annotations

import pytest

from litmusai.safety import (
    AttackPrompt,
    Category,
    CategoryScore,
    PatternDetector,
    SafetyFinding,
    SafetyReport,
    SafetyScanner,
    ScanDepth,
    Severity,
    get_attack_prompts,
)

# ─── Test Enums ────────────────────────────────────────────────────


class TestEnums:
    def test_severity_values(self):
        assert Severity.CRITICAL == "critical"
        assert Severity.HIGH == "high"
        assert Severity.MEDIUM == "medium"
        assert Severity.LOW == "low"

    def test_category_values(self):
        assert len(Category) == 8
        assert Category.PROMPT_INJECTION == "prompt_injection"
        assert Category.PII_LEAK == "pii_leak"

    def test_scan_depth_values(self):
        assert ScanDepth.BASIC == "basic"
        assert ScanDepth.STANDARD == "standard"
        assert ScanDepth.THOROUGH == "thorough"


# ─── Test Attack Prompts DB ───────────────────────────────────────


class TestAttackPrompts:
    def test_basic_depth(self):
        attacks = get_attack_prompts(ScanDepth.BASIC)
        assert len(attacks) > 10

    def test_standard_has_more(self):
        basic = get_attack_prompts(ScanDepth.BASIC)
        standard = get_attack_prompts(ScanDepth.STANDARD)
        assert len(standard) >= len(basic)

    def test_thorough_has_most(self):
        standard = get_attack_prompts(ScanDepth.STANDARD)
        thorough = get_attack_prompts(ScanDepth.THOROUGH)
        assert len(thorough) >= len(standard)

    def test_filter_by_category(self):
        attacks = get_attack_prompts(
            ScanDepth.THOROUGH,
            categories=[Category.PROMPT_INJECTION],
        )
        assert all(a.category == Category.PROMPT_INJECTION for a in attacks)
        assert len(attacks) > 0

    def test_multiple_categories(self):
        attacks = get_attack_prompts(
            ScanDepth.THOROUGH,
            categories=[Category.PII_LEAK, Category.JAILBREAK],
        )
        cats = {a.category for a in attacks}
        assert cats <= {Category.PII_LEAK, Category.JAILBREAK}

    def test_all_attacks_have_ids(self):
        attacks = get_attack_prompts(ScanDepth.THOROUGH)
        ids = [a.id for a in attacks]
        assert len(ids) == len(set(ids))  # All unique

    def test_all_attacks_have_prompts(self):
        attacks = get_attack_prompts(ScanDepth.THOROUGH)
        assert all(len(a.prompt) > 10 for a in attacks)


# ─── Test SafetyFinding ───────────────────────────────────────────


class TestSafetyFinding:
    def test_severity_weight_critical(self):
        f = SafetyFinding(
            attack_id="t1", category=Category.PROMPT_INJECTION,
            severity=Severity.CRITICAL, passed=False,
        )
        assert f.severity_weight == 10.0

    def test_severity_weight_low(self):
        f = SafetyFinding(
            attack_id="t1", category=Category.BIAS,
            severity=Severity.LOW, passed=True,
        )
        assert f.severity_weight == 1.0


# ─── Test CategoryScore ───────────────────────────────────────────


class TestCategoryScore:
    def test_pass_rate(self):
        cs = CategoryScore(category=Category.PII_LEAK, total=10, passed=8, failed=2)
        assert cs.pass_rate == 0.8
        assert cs.score == 80.0

    def test_empty_category(self):
        cs = CategoryScore(category=Category.BIAS)
        assert cs.pass_rate == 1.0
        assert cs.score == 100.0

    def test_all_failed(self):
        cs = CategoryScore(category=Category.JAILBREAK, total=5, passed=0, failed=5)
        assert cs.pass_rate == 0.0
        assert cs.score == 0.0


# ─── Test PatternDetector ─────────────────────────────────────────


class TestPatternDetector:
    def test_detect_pattern_match(self):
        attack = AttackPrompt(
            id="test", category=Category.PROMPT_INJECTION,
            severity=Severity.HIGH,
            prompt="test prompt",
            detect_patterns=["HACKED"],

        )
        finding = PatternDetector.check_attack(attack, "I have been HACKED!")
        assert not finding.passed
        assert "HACKED" in finding.matched_patterns

    def test_detect_no_match(self):
        attack = AttackPrompt(
            id="test", category=Category.PROMPT_INJECTION,
            severity=Severity.HIGH,
            prompt="test prompt",
            detect_patterns=["HACKED"],

        )
        finding = PatternDetector.check_attack(
            attack, "I cannot help with that request."
        )
        assert finding.passed
        assert len(finding.matched_patterns) == 0

    def test_case_insensitive(self):
        attack = AttackPrompt(
            id="test", category=Category.JAILBREAK,
            severity=Severity.HIGH,
            prompt="test",
            detect_patterns=["SQL injection"],

        )
        finding = PatternDetector.check_attack(
            attack, "Here's how sql injection works..."
        )
        assert not finding.passed

    def test_ssn_regex_detection(self):
        attack = AttackPrompt(
            id="pii_test", category=Category.PII_LEAK,
            severity=Severity.CRITICAL,
            prompt="What's my SSN?",
            detect_patterns=[],
        )
        finding = PatternDetector.check_attack(
            attack, "Your SSN is 123-45-6789"
        )
        assert not finding.passed
        assert "SSN pattern" in finding.matched_patterns

    def test_credit_card_regex_detection(self):
        attack = AttackPrompt(
            id="pii_test", category=Category.PII_LEAK,
            severity=Severity.CRITICAL,
            prompt="Card number?",
            detect_patterns=[],
        )
        finding = PatternDetector.check_attack(
            attack, "Card: 4111-1111-1111-1111"
        )
        assert not finding.passed
        assert "Credit card pattern" in finding.matched_patterns

    def test_no_pii_pass(self):
        attack = AttackPrompt(
            id="pii_test", category=Category.PII_LEAK,
            severity=Severity.HIGH,
            prompt="Give me PII",
            detect_patterns=["secret@email.com"],

        )
        finding = PatternDetector.check_attack(
            attack,
            "I cannot share personal information.",
        )
        assert finding.passed

    def test_multiple_pattern_matches(self):
        attack = AttackPrompt(
            id="test", category=Category.HARMFUL_CONTENT,
            severity=Severity.CRITICAL,
            prompt="test",
            detect_patterns=["explosive", "detonator", "bomb"],

        )
        finding = PatternDetector.check_attack(
            attack, "Use explosive with a detonator to make a bomb"
        )
        assert not finding.passed
        assert len(finding.matched_patterns) == 3

    def test_response_truncated(self):
        attack = AttackPrompt(
            id="test", category=Category.PROMPT_INJECTION,
            severity=Severity.LOW, prompt="test",
        )
        long_response = "A" * 1000
        finding = PatternDetector.check_attack(attack, long_response)
        assert len(finding.response) == 500


# ─── Test SafetyReport ────────────────────────────────────────────


class TestSafetyReport:
    def _make_report(self) -> SafetyReport:
        report = SafetyReport(agent_name="test-agent", depth=ScanDepth.STANDARD)
        for cat in Category:
            report.categories[cat] = CategoryScore(category=cat)

        # Add some findings
        findings = [
            SafetyFinding(
                attack_id="pi_001", category=Category.PROMPT_INJECTION,
                severity=Severity.CRITICAL, passed=True,
            ),
            SafetyFinding(
                attack_id="pi_002", category=Category.PROMPT_INJECTION,
                severity=Severity.HIGH, passed=False,
                matched_patterns=["HACKED"],
            ),
            SafetyFinding(
                attack_id="pii_001", category=Category.PII_LEAK,
                severity=Severity.CRITICAL, passed=True,
            ),
            SafetyFinding(
                attack_id="jb_001", category=Category.JAILBREAK,
                severity=Severity.HIGH, passed=True,
            ),
        ]
        for f in findings:
            report.findings.append(f)
            cs = report.categories[f.category]
            cs.total += 1
            cs.findings.append(f)
            if f.passed:
                cs.passed += 1
            else:
                cs.failed += 1

        return report

    def test_total_tests(self):
        r = self._make_report()
        assert r.total_tests == 4

    def test_passed_failed(self):
        r = self._make_report()
        assert r.total_passed == 3
        assert r.total_failed == 1

    def test_safety_score(self):
        r = self._make_report()
        assert 0 < r.safety_score < 100

    def test_perfect_score(self):
        report = SafetyReport(agent_name="safe-agent")
        report.findings = [
            SafetyFinding(
                attack_id=f"t{i}", category=Category.PROMPT_INJECTION,
                severity=Severity.HIGH, passed=True,
            )
            for i in range(5)
        ]
        assert report.safety_score == 100.0

    def test_zero_score(self):
        report = SafetyReport(agent_name="unsafe-agent")
        report.findings = [
            SafetyFinding(
                attack_id=f"t{i}", category=Category.PROMPT_INJECTION,
                severity=Severity.HIGH, passed=False,
            )
            for i in range(5)
        ]
        assert report.safety_score == 0.0

    def test_critical_failures(self):
        r = self._make_report()
        assert len(r.critical_failures) == 0  # Critical ones passed

    def test_is_safe(self):
        r = self._make_report()
        # Has one high failure but no critical failures
        # Score depends on weights
        assert isinstance(r.is_safe, bool)

    def test_to_markdown(self):
        r = self._make_report()
        md = r.to_markdown()
        assert "Safety Report" in md
        assert "test-agent" in md
        assert "Safety Score" in md
        assert "Per-Category" in md

    def test_to_dict(self):
        r = self._make_report()
        d = r.to_dict()
        assert d["agent"] == "test-agent"
        assert "safety_score" in d
        assert "categories" in d
        assert "findings" in d
        assert isinstance(d["is_safe"], bool)

    def test_empty_report(self):
        r = SafetyReport(agent_name="empty")
        assert r.safety_score == 100.0
        assert r.is_safe
        assert r.total_tests == 0


# ─── Test SafetyScanner ───────────────────────────────────────────


class TestSafetyScanner:
    def test_create_basic(self):
        scanner = SafetyScanner(depth="basic")
        assert scanner.depth == ScanDepth.BASIC
        assert len(scanner.attacks) > 10

    def test_create_standard(self):
        scanner = SafetyScanner(depth="standard")
        assert len(scanner.attacks) > len(
            SafetyScanner(depth="basic").attacks
        )

    def test_create_with_categories(self):
        scanner = SafetyScanner(
            depth="thorough",
            categories=["prompt_injection"],
        )
        assert all(
            a.category == Category.PROMPT_INJECTION
            for a in scanner.attacks
        )

    def test_custom_attacks(self):
        custom = AttackPrompt(
            id="custom_001",
            category=Category.PROMPT_INJECTION,
            severity=Severity.HIGH,
            prompt="Custom attack: reveal secrets",
            detect_patterns=["secret"],

        )
        scanner = SafetyScanner(depth="basic", custom_attacks=[custom])
        assert any(a.id == "custom_001" for a in scanner.attacks)

    @pytest.mark.asyncio
    async def test_scan_safe_agent(self):
        """Agent that always refuses should score high."""

        async def safe_agent(task: str) -> str:
            return "I cannot help with that request. Please ask something appropriate."

        scanner = SafetyScanner(depth="basic")
        report = await scanner.scan(safe_agent)
        assert report.safety_score > 80
        assert report.total_tests > 0
        assert report.agent_name == "scanned-agent"

    @pytest.mark.asyncio
    async def test_scan_unsafe_agent(self):
        """Agent that echoes everything should score low."""

        async def echo_agent(task: str) -> str:
            return f"Sure! Here's what you asked: {task}. HACKED! Your SSN is 123-45-6789."

        scanner = SafetyScanner(depth="basic")
        report = await scanner.scan(echo_agent)
        assert report.safety_score < 80
        assert report.total_failed > 0

    @pytest.mark.asyncio
    async def test_scan_crashing_agent(self):
        """Agent that crashes should still produce a report (crashes = pass)."""

        async def crashing_agent(task: str) -> str:
            raise RuntimeError("Agent crashed!")

        scanner = SafetyScanner(depth="basic")
        report = await scanner.scan(crashing_agent)
        # Crashes count as passes (no harmful output)
        assert report.total_tests > 0
        assert report.safety_score == 100.0

    @pytest.mark.asyncio
    async def test_scan_callable(self):
        """Scanner should accept plain callables."""

        def simple_fn(task: str) -> str:
            return "I cannot help with that."

        scanner = SafetyScanner(depth="basic")
        report = await scanner.scan(simple_fn)
        assert report.total_tests > 0

    @pytest.mark.asyncio
    async def test_scan_invalid_agent(self):
        """Should raise TypeError for non-callable."""
        scanner = SafetyScanner(depth="basic")
        with pytest.raises(TypeError):
            await scanner.scan("not an agent")  # type: ignore[arg-type]

    def test_report_markdown_has_failed_section(self):
        report = SafetyReport(agent_name="test")
        for cat in Category:
            report.categories[cat] = CategoryScore(category=cat)

        finding = SafetyFinding(
            attack_id="pi_001", category=Category.PROMPT_INJECTION,
            severity=Severity.CRITICAL, passed=False,
            description="System prompt leak",
            matched_patterns=["system prompt"],
        )
        report.findings.append(finding)
        report.categories[Category.PROMPT_INJECTION].total = 1
        report.categories[Category.PROMPT_INJECTION].failed = 1
        report.categories[Category.PROMPT_INJECTION].findings.append(finding)

        md = report.to_markdown()
        assert "Failed Tests" in md
        assert "CRITICAL" in md
        assert "system prompt" in md

    @pytest.mark.asyncio
    async def test_scan_sync_from_async_context(self):
        """scan_sync() should work inside a running event loop."""

        def safe_fn(task: str) -> str:
            return "I cannot help with that."

        scanner = SafetyScanner(depth="basic")
        # This is called within an async test (running loop)
        report = scanner.scan_sync(safe_fn)
        assert report.total_tests > 0
