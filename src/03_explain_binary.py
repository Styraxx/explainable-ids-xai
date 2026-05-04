import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from lime.lime_tabular import LimeTabularExplainer

DATA = os.path.join("data", "processed", "cic_ids2017_clean.parquet")
MODEL = os.path.join("outputs", "models", "rf_binary.joblib")

SHAP_SUMMARY = os.path.join("outputs", "figures", "shap_summary_binary.png")
SHAP_BAR = os.path.join("outputs", "figures", "shap_bar_binary.png")
LIME_DIR = os.path.join("outputs", "figures", "lime_cases")
ERROR_CASES = os.path.join("outputs", "tables", "error_cases.csv")

SAMPLE_SIZE = 4000  # stratified subset for SHAP/LIME (safe for 8GB RAM)

def stratified_sample(X: pd.DataFrame, y: pd.Series, n: int, seed: int = 42):
    """Return a roughly balanced stratified sample for explanation."""
    df = X.copy()
    df["y"] = y.values

    n0 = min(n // 2, int((df["y"] == 0).sum()))
    n1 = min(n - n0, int((df["y"] == 1).sum()))

    df0 = df[df["y"] == 0].sample(n=n0, random_state=seed)
    df1 = df[df["y"] == 1].sample(n=n1, random_state=seed)

    out = pd.concat([df0, df1]).sample(frac=1, random_state=seed)

    y_out = out["y"].astype(int)
    X_out = out.drop(columns=["y"])
    return X_out, y_out

def main():
    # Load data
    df = pd.read_parquet(DATA)
    y = df["label_binary"].astype(int)
    X = df.drop(columns=["label_binary", "label_original"])

    # Recreate same split params as training script
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Load model
    model = load(MODEL)

    # Predict on full test set (for FP/FN selection)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)

    # Identify indices for TP / FP / FN examples
    idx = X_test.index.to_numpy()
    yt = y_test.to_numpy()
    yp = y_pred

    tp_idx = idx[(yt == 1) & (yp == 1)][:2]
    fp_idx = idx[(yt == 0) & (yp == 1)][:2]
    fn_idx = idx[(yt == 1) & (yp == 0)][:2]

    cases = [("TP", int(i)) for i in tp_idx] + \
            [("FP", int(i)) for i in fp_idx] + \
            [("FN", int(i)) for i in fn_idx]

    # Save error case indices for evidence + repeatability
    os.makedirs(os.path.dirname(ERROR_CASES), exist_ok=True)
    pd.DataFrame(cases, columns=["case_type", "index"]).to_csv(ERROR_CASES, index=False)
    print(f"Saved error case indices -> {ERROR_CASES}")

    # Stratified sample for explanation (memory-safe)
    X_exp, y_exp = stratified_sample(X_test, y_test, n=SAMPLE_SIZE, seed=42)
    print(f"Explanation subset shape: {X_exp.shape} (balanced approx)")

    # ---------------- SHAP ----------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_exp)

    # Robustly select SHAP values for the "Attack" class
    # SHAP may return:
    # - list of arrays: [class0, class1]
    # - ndarray: (n_samples, n_features) or (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        sv = shap_values[1]  # attack class
    else:
        sv = shap_values
        # If 3D, take class 1
        if hasattr(sv, "ndim") and sv.ndim == 3 and sv.shape[2] >= 2:
            sv = sv[:, :, 1]

    # Final safety check
    if sv.shape[1] != X_exp.shape[1]:
        raise ValueError(
            f"SHAP values shape {sv.shape} does not match data shape {X_exp.shape}. "
            f"Check SHAP output format / feature alignment."
        )

    os.makedirs(os.path.dirname(SHAP_SUMMARY), exist_ok=True)

    # Summary plot
    plt.figure()
    shap.summary_plot(sv, X_exp, show=False)
    plt.savefig(SHAP_SUMMARY, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot -> {SHAP_SUMMARY}")

    # Bar plot
    plt.figure()
    shap.summary_plot(sv, X_exp, plot_type="bar", show=False)
    plt.savefig(SHAP_BAR, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP bar plot -> {SHAP_BAR}")

    # ---------------- LIME ----------------
    os.makedirs(LIME_DIR, exist_ok=True)

    # Use a small sample from training set for LIME explainer background
    background = X_train.sample(5000, random_state=42).to_numpy()

    lime = LimeTabularExplainer(
        training_data=background,
        feature_names=X_train.columns.tolist(),
        class_names=["Benign", "Attack"],
        mode="classification"
    )

    for case_type, row_index in cases:
        x_row = X_test.loc[row_index].to_numpy()
        exp = lime.explain_instance(
            data_row=x_row,
            predict_fn=model.predict_proba,
            num_features=10
        )
        out_html = os.path.join(LIME_DIR, f"lime_{case_type}_{row_index}.html")
        exp.save_to_file(out_html)
        print(f"Saved LIME explanation -> {out_html}")

    print("Done: SHAP + LIME explanations generated successfully.")

if __name__ == "__main__":
    main()