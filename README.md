# Women's E-Commerce Reviews

Multimodal ML: **68+ engineered features** (NLP, vision-language, image CV, research survey) + TF-IDF → logistic regression. Predicts whether a customer will recommend a clothing item.

**Dataset:** [Kaggle — Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

Place `Womens Clothing E-Commerce Reviews.csv` in `data/raw/`.

## Commands

```bash
make install   # pip install dependencies
make train     # train model → models/model.joblib
make app       # ReviewSense UI — predict, explore data, batch CSV (http://localhost:8501)
```

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
