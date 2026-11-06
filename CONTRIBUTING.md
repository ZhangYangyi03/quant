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

No API key or special hardware is needed for the checks below (standard library
plus, where noted, pytest):

```bash
python -B -m unittest discover -s tests -v
python -B run_institutional.py --codes 20 --days 200 --out /tmp/institutional_run.json
```

CI runs the unit suite on Python 3.11 and 3.13 and the six-gate institutional demo on synthetic data. Two tests skip when the 1.9 GB external dataset is absent; see DATA.md for the reconstruction chain.

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
