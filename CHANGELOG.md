# Changelog

All notable changes are documented here. This file starts with the
2026-11 GitHub portfolio audit: the fixes below were made in response to
`GitHub-项目综合评价.md` and are intentionally itemised so a reviewer can
verify each one against the commit.

## [Unreleased] — 2026-11 portfolio audit fixes

- Added MIT LICENSE (the repository previously had none).
- Added .github/workflows/ci.yml: 110-test suite on Python 3.11 / 3.13 plus the six-gate synthetic demo.
- Added CI and license badges and documented what CI actually checks.
- Fixed the first red CI run: the workflow assumed the whole suite was standard-library only, but `tests/test_strategy_layer.py` imports `strategy_layer`, which imports numpy at module level. CI now installs the declared dependencies before the suite, and its "no third-party package" check imports the real core modules instead of asserting on a fresh interpreter (which proved nothing).
- Relaxed `requirements.txt` from `numpy<2.0` to `numpy>=1.24`. The old upper bound has no wheel on Python 3.13; the 110-test suite passes on numpy 2.5.4.

## Audit context

- Full portfolio assessment: `GitHub-项目综合评价.md` (overall 6.8/10; external validation 1.5/10).
- Canonical maintainer identity: `ZhangYangyi03` / Yangyi Zhang.
- This changelog does not claim external validation. See the "Independent reproduction" issue template.
