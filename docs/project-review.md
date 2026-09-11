# Project review

Reviewed against `7a38b4a` on September 10, 2026. The recommendations below concern implementation behavior, rather than whether a model performs well on the bundled suites.

## Changes prepared in this review

- **Failed scans:** API failures previously counted as successful safety checks, and conversation steps without assertions passed even when the agent failed. Failed calls now remain failed, with error details and inconclusive scan verdicts.
- **GitHub Action:** Inputs were interpolated into shell and Python source, paths were split on spaces, and declared outputs were not mapped to step outputs. The action now passes argument lists, forwards outputs, clears stale result files, and installs the checked-out action version.
- **Windows support:** The baseline had 31 failures on Windows. Reports used the system encoding despite containing Unicode, timing used a coarse clock on Python 3.11, and subprocess and temporary-file tests assumed Unix behavior. The portability changes pass the existing suite and add Windows CI and a Unicode report check.
- **CLI behavior:** Starter assertions used an unsupported `values` key and checked an empty string. Profile run counts and concurrency were discarded during configuration merging. The fixes use valid assertion fields and preserve explicit CLI settings over profiles, and profiles over config defaults. The version flag now uses the package version.
- **Documentation and presentation:** Removed decorative emojis, unsupported model rankings, untraceable research claims, and claims that costs come directly from billing responses. Added a runnable local quick start and a check for emojis in new PR titles and commit messages. Existing commits and their dates remain unchanged.

## Recommended next work

### 1. Use all runs for CI gates and budgets

`src/litmusai/ci/__init__.py:run_evaluation` uses only the last run for threshold and budget checks when `runs > 1`. `Pipeline` uses the first run for its primary result and threshold. A flaky agent can pass depending on run order, and repeated API charges are omitted from the CLI budget comparison.

Use aggregate pass rate and total cost for gates. Retain per-run results in exported JSON, and define how a multi-run baseline compares with a single-run baseline. Add tests where the first and last runs disagree and where each run is under budget but the total exceeds it.

### 2. Make adapter configuration consistent

`Agent.from_openai_chat` uses only its explicit API key and URL; it does not resolve the global configuration advertised in earlier examples. The OpenAI Agents SDK adapter also assumes imports and response fields without integration coverage for the current SDK.

Share key, URL, and header resolution across adapters. Test outgoing requests using mocked HTTP responses, and test optional SDK adapters against supported SDK versions. Keep those versions documented.

### 3. Unify result formats

`EvalResults.to_dict()` and `ci.results_to_dict()` emit different shapes. The CLI schema omits stable case IDs and model parameters, while diffing and other consumers expect the richer schema. Multi-run CLI JSON also discards the per-run statistics.

Define one versioned result schema, use it for CLI and Python exports, and keep a reader for existing files. Test a complete CLI run-to-diff/report workflow so metadata loss is visible.

### 4. Validate evaluation inputs before scheduling work

The Python runners accept zero concurrency. `evaluate(..., concurrency=0, verbose=False)` waits indefinitely on a semaphore; the verbose path can instead return no results. Empty suites and scans can also yield misleading status values.

Validate counts in the public Python APIs as well as the CLI, and define an explicit result for evaluations with no executed cases. Include malformed configuration and empty-suite cases in tests.

### 5. Separate measured behavior from heuristics

Safety checks use pattern matching; a refusal phrase can suppress a matched attack pattern. Conversation cascade detection treats every failure after the first as a cascade. Neither mechanism establishes what the model intended or why a later step failed.

Expose the evidence behind a finding, document the heuristic, and add examples that include both a refusal and harmful content. Prefer an explicit unknown state when the available evidence cannot support a conclusion.

## Validation

The baseline passed lint and strict type checking. On Windows, 808 tests passed, 31 failed, and 7 provider integration tests were skipped. The portability branch passed 840 tests after its changes. All five branches merged cleanly in a local integration checkout: 867 tests passed, 7 were skipped, lint and strict type checking passed, and both source and wheel packages built. New regressions cover failed scans, literal action inputs and exit codes, configuration precedence, useful starter assertions, and commit-style checks that exclude inherited history.

Provider integration tests were not run against live services. No published Git history was rewritten.
