# Contributing to extensions/

This document covers how to make safe edits, run tests, and add new submodules under `extensions/`.

Basic rules

- Keep changes to source files minimal and well-tested; add unit tests under each submodule's `tests/` directory.
- Avoid changing archived/backup files unless you are intentionally cleaning up history.

Local development

- Recommended workflow:

  1. Create a feature branch off main:
     git checkout -b feat/extensions-xxxxx

  2. Make changes and add tests under the corresponding submodule (`extensions/custom_mean/tests`, etc.).

  3. Run local smoke tests. The repository includes some helper scripts in `tools/` (e.g. `tools/check_imports.py`). You may run tests using `pixi` in this environment.

  4. Run limited test battery for the affected folder(s). For example, to run tests in a submodule:
     pixi run pytest -- extensions/custom_mean/tests

Testing and CI

- CI should be configured to run tests for changed areas only to conserve resources. If you'd like, I can scaffold GitHub Actions workflows to run per-submodule tests.

Style and docs

- Keep docs focused and accurate. Use `extensions/docs/` as the single source-of-truth for extension-level information.
- For long-form guides or API explanations, put them under `extensions/docs/` or `extensions/docs/top_level/` (if you later need a separate top-level docs grouping).
