#!/bin/bash
# Run: GITHUB_TOKEN=ghp_your_token bash cleanup_issues.sh
# Or: gh auth login first, then: bash cleanup_issues.sh

TOKEN="${GITHUB_TOKEN:-$(gh auth token 2>/dev/null)}"
REPO="kutanti/litmusai"
API="https://api.github.com/repos/$REPO/issues"
H=(-H "Authorization: token $TOKEN" -H "Accept: application/vnd.github+json")

echo "=== CLOSING 7 shipped/duplicate issues ==="

for issue in 21 23 32 36 38 41 43; do
  echo -n "Closing #$issue... "
  curl -s -X PATCH "${H[@]}" "$API/$issue" \
    -d '{"state":"closed","state_reason":"completed"}' | python3 -c "import sys,json; print(json.load(sys.stdin).get('state','error'))"
done

echo ""
echo "=== CLOSING #37 (merged into #27) ==="
curl -s -X POST "${H[@]}" "$API/37/comments" \
  -d '{"body":"Merged into #27 — both issues ask for academic benchmark sourcing."}' > /dev/null
curl -s -X PATCH "${H[@]}" "$API/37" \
  -d '{"state":"closed","state_reason":"not_planned"}' | python3 -c "import sys,json; print(json.load(sys.stdin).get('state','error'))"

echo ""
echo "=== RENAMING 4 issues ==="

# #26
curl -s -X PATCH "${H[@]}" "$API/26" \
  -d '{"title":"🎯 Log model params in results + benchmark defaults","body":"## Problem\nBenchmarks don'\''t control or log model parameters. Without standardization, comparisons aren'\''t fair.\n\n## What exists\n- `Agent.from_openai_chat()` already takes `temperature`, `max_tokens`\n- Results JSON logs scores and cost\n\n## What'\''s needed\n1. Default `temperature=0` when running built-in benchmark suites\n2. Log all model params (`temperature`, `max_tokens`, `model`, `seed`) in results JSON\n3. Warn if `temperature > 0` in benchmark mode\n4. Add `--seed` flag to CLI for reproducibility\n\n## Priority\n**Medium** — improves reproducibility"}' > /dev/null && echo "#26 updated" || echo "#26 failed"

# #27 (absorbing #37)
curl -s -X PATCH "${H[@]}" "$API/27" \
  -d '{"title":"📈 Source academic benchmarks (GSM8K, HumanEval, TruthfulQA)","body":"## Problem\nBuilt-in suites have 50 cases across 8 domains. For credible benchmarks, we need 50+ per domain sourced from established academic datasets.\n\n## Current state\n8 suites, ~50 total cases (coding, research, safety, planning, customer_support, summarization, instruction_following, tool_use)\n\n## What'\''s needed\n1. Source from GSM8K (math), HumanEval (code), MMLU subsets, TruthfulQA\n2. Proper attribution and licensing\n3. Difficulty levels (easy/medium/hard)\n4. Ground truth with citations\n\n*Subsumes #37 (Curated Benchmark Library)*\n\n## Priority\n**Low** — nice-to-have, not blocking adoption"}' > /dev/null && echo "#27 updated" || echo "#27 failed"

# #35
curl -s -X PATCH "${H[@]}" "$API/35" \
  -d '{"title":"🔄 Pipeline convenience class — chain eval + safety + report","body":"## Problem\nRunning a full evaluation (run tasks → assert → score → safety scan → generate report) requires gluing multiple function calls together.\n\n## What exists\n- `evaluate()` — runs tasks with assertions, scoring, cost tracking ✅\n- `SafetyScanner.scan()` — safety scanning ✅\n- `render_html()` — HTML report generation ✅\n- `to_junit_xml()` / `to_csv()` — export ✅\n- `litmus run` CLI — ties it together ✅\n\n## What'\''s needed\nA `Pipeline` class that chains everything in one call:\n\n```python\nfrom litmusai import Pipeline\n\npipeline = Pipeline(\n    agent=my_agent,\n    suite=\"coding\",\n    safety=True,\n    report=\"html\",\n    runs=3,\n)\nresults = await pipeline.run()\n```\n\nThis is a convenience wrapper, NOT a rewrite. All pieces exist.\n\n## Priority\n**Medium** — improves DX"}' > /dev/null && echo "#35 updated" || echo "#35 failed"

# #44
curl -s -X PATCH "${H[@]}" "$API/44" \
  -d '{"title":"🔬 CLI evaluation profiles (--profile quick|thorough|benchmark)","body":"## Problem\nUsers manually configure scorer, judge, safety, concurrency for every eval run.\n\n## What exists\n- `litmus init` — scaffolds config file ✅\n- `.litmus/config.yaml` — config file support ✅\n- All eval features available via CLI flags ✅\n\n## What'\''s needed\nPreset profiles that set sensible defaults:\n\n```bash\nlitmus run -s coding -a my_agent --profile quick\n# → concurrency=10, no LLM judge, no safety, 1 run\n\nlitmus run -s coding -a my_agent --profile thorough  \n# → concurrency=3, LLM judge, safety scan, 3 runs, HTML report\n\nlitmus run -s coding -a my_agent --profile benchmark\n# → temperature=0, 5 runs, full logging, reproducible\n```\n\n## Priority\n**Medium** — improves onboarding"}' > /dev/null && echo "#44 updated" || echo "#44 failed"

echo ""
echo "=== DONE ==="
echo "Final open issues: #26, #27, #28, #29, #33, #34, #35, #44 (8 total)"
