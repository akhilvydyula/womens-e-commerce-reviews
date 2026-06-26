# Women's E-Commerce Reviews

[![License: MIT](https://img.shields.io/github/license/akhilvydyula/womens-e-commerce-reviews)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![GitHub stars](https://img.shields.io/github/stars/akhilvydyula/womens-e-commerce-reviews?style=social)](https://github.com/akhilvydyula/womens-e-commerce-reviews/stargazers)
[![Open Source](https://img.shields.io/badge/open%20source-welcome-brightgreen)](#open-source)

Multimodal ML: **68+ engineered features** (NLP, vision-language, image CV, research survey) + TF-IDF → logistic regression. Predicts whether a customer will recommend a clothing item.

**Dataset:** [Kaggle — Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

Place `Womens Clothing E-Commerce Reviews.csv` in `data/raw/`.

## Commands

```bash
make install   # pip install dependencies
make train     # train model → models/model.joblib
make app       # ReviewSense UI — predict, explore data, batch CSV (http://localhost:8501)
```

## Deploy on Render

Use the following values in Render's "New Web Service" form:

- **Language:** `Python 3`
- **Branch:** `main`
- **Root Directory:** *(leave blank)*
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run src/app.py --server.headless true --server.address 0.0.0.0 --server.port $PORT`

This repo includes a `render.yaml` blueprint with the same settings.

Important for full prediction features:
- The app expects a trained model at `models/model.joblib`.
- If the model file is not present in your deployed code, set Render env var `MODEL_URL` to a public URL of `model.joblib`.
- Example Render env var:
  - `MODEL_URL=https://<public-host>/model.joblib`
- You can also train locally (`make train`) and publish that artifact to a public object URL.

## Project layout

```text
data/raw/          # Kaggle CSV
models/            # model.joblib (after train)
src/
  app.py              # Streamlit research UI
  train.py            # training
  inference.py        # multimodal prediction
  nlp_features.py     # 25 lexical NLP features
  vision_features.py  # vision-language + image CV
  survey.py           # 24-question protocol
  feature_engineering.py
  ui_survey.py        # questionnaire widgets
  data.py / features.py / config.py
```

## Open source

This repository is **open source** under the [MIT License](LICENSE). Stars, issues, and pull requests are welcome — they help others discover the project and improve it for the community.

### Project docs for contributors

| Topic | Document |
|-------|----------|
| Full OSS inventory (CI/CD, security, community) | [docs/OPEN_SOURCE.md](docs/OPEN_SOURCE.md) |
| How to contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Security policy | [SECURITY.md](SECURITY.md) |
| Compliance and data governance | [docs/COMPLIANCE.md](docs/COMPLIANCE.md) |
| Code of conduct | [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |

### How you can help

- **Star** the repo if you find it useful — it helps visibility on GitHub Explore and search.
- **Open an issue** for bugs, ideas, or questions.
- **Submit a pull request** with a focused change and a clear description.
- **Share** the project with anyone learning NLP, multimodal features, or e-commerce analytics.

Maintained by [Akhil Vydyula](https://github.com/akhilvydyula) as part of the Skills Marathon ML portfolio.

## License

Application code is released under the [MIT License](LICENSE). The [Kaggle dataset](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) has its own terms — review before redistribution.
