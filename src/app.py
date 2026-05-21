"""Streamlit app — run with: make app"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import CATEGORICAL_COLUMNS, MODEL_PATH, RAW_FILE_PATH, TARGET_COLUMN
from src.data import load_raw_data
from src.inference import load_model, predict

st.set_page_config(page_title="Review Recommender", layout="centered")
st.title("Will the customer recommend this item?")

try:
    model = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

cat_opts = {col: ["Unknown"] for col in CATEGORICAL_COLUMNS}
if RAW_FILE_PATH.exists():
    df = load_raw_data()
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            cat_opts[col] = sorted(df[col].dropna().astype(str).unique().tolist())

with st.form("review"):
    title = st.text_input("Title")
    review = st.text_area("Review", height=120)
    rating = st.slider("Rating", 1.0, 5.0, 5.0, 0.5)
    age = st.number_input("Age", 0.0, 120.0, 35.0)
    feedback = st.number_input("Positive feedback count", 0, step=1)
    division = st.selectbox("Division", cat_opts["Division Name"])
    department = st.selectbox("Department", cat_opts["Department Name"])
    class_name = st.selectbox("Class", cat_opts["Class Name"])
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    if not review.strip() and not title.strip():
        st.warning("Enter a title or review.")
    else:
        row = {
            "Title": title,
            "Review Text": review,
            "Age": age,
            "Positive Feedback Count": feedback,
            "Rating": rating,
            "Division Name": division,
            "Department Name": department,
            "Class Name": class_name,
        }
        pred, score = predict(model, row)
        if pred == 1:
            st.success(f"Recommended — confidence {score:.0%}")
        else:
            st.error(f"Not recommended — confidence {1 - score:.0%}")

if RAW_FILE_PATH.exists() and st.button("Try a random review"):
    sample = load_raw_data().sample(1).iloc[0]
    row = {
        "Title": str(sample.get("Title", "")),
        "Review Text": str(sample.get("Review Text", "")),
        "Age": float(sample.get("Age", 0) or 0),
        "Positive Feedback Count": int(sample.get("Positive Feedback Count", 0) or 0),
        "Rating": float(sample.get("Rating", 5) or 5),
        "Division Name": str(sample.get("Division Name", "Unknown")),
        "Department Name": str(sample.get("Department Name", "Unknown")),
        "Class Name": str(sample.get("Class Name", "Unknown")),
    }
    st.write(f"**{row['Title']}**")
    st.write(row["Review Text"][:400])
    if TARGET_COLUMN in sample.index:
        actual = "Recommended" if int(sample[TARGET_COLUMN]) == 1 else "Not recommended"
        st.caption(f"Actual: {actual}")
    pred, score = predict(model, row)
    st.info(
        f"Prediction: {'Recommended' if pred == 1 else 'Not recommended'} ({score:.0%})"
    )
