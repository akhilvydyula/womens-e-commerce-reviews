# Women's E-Commerce Reviews

Predict whether a customer will recommend a clothing item from their review.

**Dataset:** [Kaggle — Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

Place `Womens Clothing E-Commerce Reviews.csv` in `data/raw/`.

## Commands

```bash
make install   # pip install dependencies
make train     # train model → models/model.joblib
make app       # open Streamlit UI (http://localhost:8501)
```

## Project layout

```text
data/raw/          # Kaggle CSV
models/            # model.joblib (after train)
src/
  app.py           # Streamlit UI
  train.py         # training
  inference.py     # prediction helpers
  data.py          # load & clean data
  features.py      # sklearn pipeline
  config.py        # paths & column names
```
