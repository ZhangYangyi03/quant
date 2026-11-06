# Contributing to quant

Thanks for looking. The single most valuable contribution to this repository is
**running the checks on your own machine and telling us what happened** —
especially if it failed. This portfolio has strong internal verification and
almost no external reproduction; an independent result, positive or negative,
is worth more than a star.

## Ground rules

- MIT licensed. By opening a pull request you agree your contribution is under the same license.
- Bug reports, reproduction attempts and arguments about the claims are contributions. You do not need to write code.
- One change per pull request.
- If a number in the README disagrees with the code, that is a bug. Please open an issue rather than editing the prose to match.

## What this repository is

An event-driven A-share research codebase with point-in-time data, T+1/limit/suspension execution rules, config fingerprints and no-lookahead acceptance gates.

## Reproduce it

The reproducible core is standard-library only; the full 110-test suite also
needs numpy for the legacy `strategy_layer` regression tests:

```bash
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
python -B run_institutional.py --codes 20 --days 200 --out /tmp/institutional_run.json
```

The third command needs no third-party package at all. CI asserts this *before*
installing anything by importing the core quantlab modules and checking that
numpy/pandas never enter `sys.modules`; only then does it install the declared
dependencies for the legacy test layer. CI runs the unit suite on Python 3.11 and
3.13 plus the six-gate institutional demo on synthetic data. Two tests skip when
the 1.9 GB external dataset is absent; see DATA.md for the reconstruction chain.

## Reporting an independent reproduction

Use the issue template "Independent reproduction". What makes a report actionable:

1. The commit hash or version you checked out.
2. `python -c "import sys, platform; print(sys.version, platform.platform())"`.
3. The exact commands and their full output (a failed check with a traceback is a good report).
4. Expected vs actual, in one sentence.
5. Whether all checks passed, some failed, numbers differed, or you could not run them.

Both "all checks passed on Ubuntu 24.04 / Python 3.12" and "check 3 fails on my
machine" are useful. Please do not fix a failing check silently in a PR — report
it first so the discrepancy is recorded.

## Claim scope

Where a document says a claim is exploratory, a working hypothesis, or not
human-adjudicated, please hold us to it. A pull request that strengthens a claim
without new evidence will be asked for the evidence.
