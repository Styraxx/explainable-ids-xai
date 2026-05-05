# Explainable Intrusion Detection System (IDS)

This project develops a machine learning-based intrusion detection system using the CIC-IDS2017 dataset.

It combines high-performance classification with explainability techniques (SHAP and LIME), and introduces an error-based explainability framework.

## Structure

- src/ → scripts for preprocessing, training, and explainability
- outputs/ → figures and tables used in the report
- docs/ → report and supporting document

## How to Run

pip install -r requirements.txt

python src/02_train_eval_binary.py

## Data

Dataset not included due to size.

Download CIC-IDS2017 and place in:
data/raw/cic_ids2017/
