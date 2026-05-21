PYTHON ?= python

.PHONY: install train app

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m src.train

app:
	$(PYTHON) -m streamlit run src/app.py
