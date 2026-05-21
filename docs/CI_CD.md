# CI/CD (GitLab)

Pipeline definition: `.gitlab-ci.yml` (GitLab runner with Docker executor).

## What runs in CI

| Layer | Tool | Purpose |
|--------|------|--------|
| Secrets | [Gitleaks](https://github.com/gitleaks/gitleaks) | Detect API keys and tokens in git history |
| Dependencies | [pip-audit](https://pypi.org/project/pip-audit/) | CVE scan on `requirements.txt` and `requirements_train.txt` |
| Python SAST | [Bandit](https://github.com/PyCQA/bandit) | Static analysis on `src/` (`-ll` = medium+ severity) |
| Tests | `unittest` + `coverage` | Regression tests; fails if coverage &lt; 70% |

**Stages:** `security` (gitleaks, pip-audit, bandit) → `test` (unittest + coverage artifacts).

Set `GIT_DEPTH: "0"` so secret scanning sees full history.

## First push to GitLab

1. Create a project and add this repo as `origin`.
2. Push a branch; open **CI/CD → Pipelines**.
3. On failure, open the failed job log. Download artifacts (`coverage.xml`, `htmlcov/`) if needed.

Store secrets (e.g. Kaggle API) in **Settings → CI/CD → Variables** (masked), never in YAML.

## Local parity

```bash
make ci-local
```

Or manually:

```bash
pip install -r requirements-ci.txt
python -m bandit -r src -ll -f txt
pip-audit -r requirements.txt
python -m coverage run -m unittest discover -s tests -p "test_*.py"
python -m coverage report -m --fail-under=70
```

Gitleaks is CI-only locally unless you install the CLI. False positives: `.gitleaksignore`.

## Files

| File | Role |
|------|------|
| `.gitlab-ci.yml` | Pipeline |
| `requirements-ci.txt` | CI dependencies |
| `pyproject.toml` | Bandit excludes |
| `.gitleaksignore` | Gitleaks tuning |

Do not commit `kaggle.json`, API keys, or `models/*.joblib` (gitignored).
