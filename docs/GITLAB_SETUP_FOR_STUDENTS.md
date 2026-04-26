# GitLab: push your project, run CI/CD, and read the pipeline (student guide)

This repo includes **`.gitlab-ci.yml`**. After you push to GitLab, GitLab automatically looks for that file and **creates a pipeline** for your branch — you do **not** need to click “create pipeline” manually for normal pushes.

---

## 1. One-time: create the project on GitLab

1. Log in to GitLab.
2. **New project** → **Create blank project** (or “Import from GitHub” if you mirror).
3. Name it (e.g. `womens-ecommerce-reviews`).
4. Do **not** initialize with a README if you already have one locally (avoids merge conflicts).

---

## 2. Push this repository from your laptop

From your project folder (where `Makefile` and `.gitlab-ci.yml` live):

```bash
git remote add origin https://gitlab.com/<your-group>/<your-project>.git
git branch -M main
git push -u origin main
```

(Use SSH if your course uses SSH keys: `git@gitlab.com:...`)

After the push, open GitLab → **Build** → **Pipelines**. You should see a **new pipeline** running.

---

## 3. What the pipeline does (high level)

| Stage | Jobs | What it checks |
|--------|------|----------------|
| **security** | `gitleaks`, `pip-audit` (2 files), `bandit` | Secrets in git, vulnerable packages, common Python security issues |
| **test** | `unittest` | Unit tests + coverage report + Cobertura artifact |

If any job **fails**, the pipeline is **red**. Click the failed job → read the **log** at the bottom (same idea as GitHub Actions).

---

## 4. Runners (why a pipeline might stay “pending”)

GitLab needs a **runner** to execute jobs.

- **GitLab.com** often provides **shared runners** (enabled by default for many namespaces).
- If jobs never start: **Settings** → **CI/CD** → **Runners** → ensure **shared runners** are enabled, or add your own **self-hosted runner**.

Your instructor may give you a **group runner** token — follow their lab doc.

---

## 5. CI/CD vs “deployment”

| Concept | In this course repo |
|---------|---------------------|
| **CI** (continuous integration) | Every push runs tests + security checks. |
| **CD** (continuous delivery/deploy) | **Not** fully automated here. You **can** add a `deploy` stage later (Docker image, Kubernetes, Databricks job). |

So: **pushing to GitLab builds the *pipeline*** (CI). **Production deployment** is a separate step you design (API container, cloud VM, Databricks job, etc.).

---

## 6. Secrets and variables (don’t put passwords in YAML)

For API keys or Kaggle tokens:

1. GitLab → **Settings** → **CI/CD** → **Variables**.
2. Add a variable (e.g. `KAGGLE_USERNAME`), mark **Masked** / **Protected** as appropriate.
3. Use it in `.gitlab-ci.yml` as `$KAGGLE_USERNAME`.

Never commit `kaggle.json` or API keys to the repo — **gitleaks** in CI is there to help catch mistakes.

---

## 7. Branch workflow (lower vs higher environment)

Typical teaching / industry pattern:

- **`develop`** or **`dev`** — integrate work; pipeline runs on every push.
- **`main`** — stable; optional **protected branch** + “pipeline must pass” before merge.

In GitLab: **Settings** → **Repository** → **Protected branches** to require passing pipeline before merge.

---

## 8. Debugging checklist

| Symptom | What to try |
|---------|-------------|
| Pipeline **pending** forever | Runner not available → enable shared runner or register runner. |
| **gitleaks** failed | Secret-like string in commit history → remove/rotate secret, or tune `.gitleaksignore` (only if false positive). |
| **pip-audit** failed | Vulnerable package → upgrade version in `requirements.txt` / `requirements_train.txt`. |
| **bandit** failed | Medium/high issue in `src/` → fix code or discuss with instructor. |
| **unittest** failed | Run `make test` locally; fix failing test or code. |

---

## 9. Local API (FastAPI) — common confusion

Running `make api` starts the server at `http://127.0.0.1:8000/`.

- **`/`** — short JSON “menu” (health, docs, predict).
- **`/docs`** — **Swagger UI** (best place to try **POST /predict**).
- **`/health`** — process check + model path.
- **`GET /` returning 404** was the old behavior; the app now serves **`/`** on purpose.

Train a model first (`make train-better`) so `models/better_pipeline.joblib` exists, or set **`MODEL_PATH`** to another artifact.

---

## 10. Related docs

- `docs/CI_CD_GITHUB_GITLAB.md` — full tool list (Gitleaks, pip-audit, Bandit) + GitHub parity
- `docs/DEPLOYMENT_RUNBOOK.md` — validate → train → inference → API
- `docs/ML_PRODUCT_RAW_TO_PRODUCTION.md` — bigger picture (data → prod → monitoring)
