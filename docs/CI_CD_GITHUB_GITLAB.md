# CI/CD: GitHub Actions + GitLab CI (security, quality, tests)

This repository includes **two** pipeline definitions so you can use the same checks with **GitHub** or **GitLab** (or both if you mirror).

---

## What runs in CI

| Layer | Tool | Purpose |
|--------|------|--------|
| **Secrets / credentials** | [Gitleaks](https://github.com/gitleaks/gitleaks) | Detect API keys, tokens, private keys accidentally committed (good for partner handoffs). |
| **Dependency vulnerabilities** | [pip-audit](https://pypi.org/project/pip-audit/) | Flags known CVEs in pinned/declared dependencies (`requirements.txt` and `requirements_train.txt`). |
| **Python SAST** | [Bandit](https://github.com/PyCQA/bandit) | Static analysis for common security mistakes in Python (`src/`). Uses **`-ll`** (medium+ severity) so intentional low-severity findings (e.g. fixed `subprocess` for Kaggle CLI) do not block merges. |
| **Tests + coverage** | `unittest` + `coverage` | Regression tests; fails if line coverage **&lt; 70%** (adjust in workflow / `.gitlab-ci.yml`). |

This is **not** a full enterprise AppSec program. It is a **teaching-grade, partner-friendly baseline** you can extend (Snyk, CodeQL, Trivy, container scan, DAST, etc.).

---

## GitHub Actions

- **File:** `.github/workflows/ci.yml`
- **Triggers:** `push` and `pull_request` to `main`, `master`, or `develop` (edit branches as needed).

**Local parity (before pushing):**

```bash
pip install -r requirements-ci.txt
python -m bandit -r src -ll -f txt
pip-audit -r requirements.txt
python -m coverage run -m unittest discover -s tests -p "test_*.py"
python -m coverage report -m --fail-under=70
```

**Debugging failed jobs:** open the workflow run → expand the failed step; logs include tool output. For `gitleaks`, if you have a **false positive**, add a pattern to `.gitleaksignore` or move secrets to **GitHub Actions secrets** / **GitLab CI variables** (masked).

**Optional:** enable **Dependabot** (`.github/dependabot.yml`) for automated dependency PRs.

---

## GitLab CI

**Student walkthrough (push → pipeline → debug):** see `docs/GITLAB_SETUP_FOR_STUDENTS.md`.

- **File:** `.gitlab-ci.yml`
- **Needs:** A **runner** with the **Docker** executor (shared SaaS runner or your own).
- **Variable:** `GIT_DEPTH: "0"` for a full clone so secret scanning sees history (trade-off: slightly slower).

**Stages**

1. `security` — `gitleaks`, `pip-audit` (×2 requirement files), `bandit`
2. `test` — install `requirements-ci.txt`, unittest, coverage XML + HTML artifacts

**Coverage in GitLab:** The job uploads **Cobertura** `coverage.xml` for the MR **Coverage** widget (GitLab version dependent).

**Debugging:** CI/CD → Pipelines → failed job → job log. Download **artifacts** (`coverage.xml`, `htmlcov/`) for local inspection.

**GitLab Ultimate (optional):** Uncomment the `include:` templates at the top of `.gitlab-ci.yml` for additional built-in SAST/dependency/secret jobs (license may apply).

---

## Sensitive data & “PA” / defect prevention

- **Do not commit:** `kaggle.json`, API keys, connection strings, production CSVs with PII. Use `.gitignore` + CI secret scan.
- **Principle of least privilege:** CI uses read-only `contents: read` on GitHub where possible.
- **Change impact:** Keep PRs small; run `make check` / `make test-cov` locally; CI enforces tests and scans on every push.
- **Model artifacts:** `models/*.joblib` are gitignored; store artifacts in your registry (MLflow, Azure ML, etc.), not in git.

---

## Reuse across partners

1. **Same repo, same YAML** — partners fork or get read access; pipelines travel with the repo.
2. **Branch protection:** require CI green before merge (`main` / `develop`).
3. **Environment secrets:** use GitHub **Environments** or GitLab **protected variables** for staging/prod credentials — never inline in YAML.

---

## Files reference

| File | Role |
|------|------|
| `requirements-ci.txt` | Lighter install + `bandit`, `pip-audit`, `coverage` for agents |
| `pyproject.toml` | `[tool.bandit]` excludes (tests, venv, notebooks, etc.) |
| `.gitleaksignore` | Tune false positives |
| `.github/workflows/ci.yml` | GitHub pipeline |
| `.gitlab-ci.yml` | GitLab pipeline |

---

## Makefile shortcut

```bash
make ci-local
```

(Added to `Makefile` — runs audit + bandit + tests + coverage gate without gitleaks, for quick local checks.)
