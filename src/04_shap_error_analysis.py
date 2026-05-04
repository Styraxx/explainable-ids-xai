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

DATA = os.path.join("data", "processed", "cic_ids2017_clean.parquet")
MODEL = os.path.join("outputs", "models", "rf_binary.joblib")

OUT_TABLE = os.path.join("outputs", "tables", "shap_error_comparison.csv")
OUT_TOPK = os.path.join("outputs", "tables", "shap_top10_by_category.csv")
OUT_OVERLAP = os.path.join("outputs", "tables", "shap_top10_overlap.csv")
FIG_DIR = os.path.join("outputs", "figures", "shap_error_analysis")

SAMPLE_SIZE = 4000          # keep consistent with your successful run
TOPK = 10
TOPPLOT = 15

def stratified_sample_indices(X: pd.DataFrame, y: pd.Series, n: int, seed: int = 42):
    """Return indices for a roughly balanced sample from X based on y."""
    idx0 = y[y == 0].sample(n=min(n // 2, int((y == 0).sum())), random_state=seed).index
    idx1 = y[y == 1].sample(n=min(n - len(idx0), int((y == 1).sum())), random_state=seed).index
    idx = pd.Index(idx0.tolist() + idx1.tolist())
    return idx.to_series().sample(frac=1, random_state=seed).index  # shuffled

def compute_shap_attack_class(explainer, X_subset: pd.DataFrame):
    """Compute SHAP values and return a 2D array aligned to X_subset for attack class."""
    shap_values = explainer.shap_values(X_subset)

    # SHAP may return list or ndarray; normalize to 2D (n_samples, n_features)
    if isinstance(shap_values, list):
        sv = shap_values[1]  # attack class
    else:
        sv = shap_values
        if hasattr(sv, "ndim") and sv.ndim == 3 and sv.shape[2] >= 2:
            sv = sv[:, :, 1]  # take attack class

    if sv.shape[0] != X_subset.shape[0] or sv.shape[1] != X_subset.shape[1]:
        raise ValueError(f"SHAP shape {sv.shape} does not match X shape {X_subset.shape}.")

    return sv

def topk_features(mean_abs: pd.Series, k: int = 10):
    return mean_abs.sort_values(ascending=False).head(k)

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))

def main():
    # Load cleaned dataset
    df = pd.read_parquet(DATA)
    y = df["label_binary"].astype(int)
    X = df.drop(columns=["label_binary", "label_original"])
    feature_names = X.columns.tolist()

    # Recreate same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Load model
    model = load(MODEL)

    # Predict on full test set to define TP/FP/FN/TN
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)

    # Stratified sample from test set for SHAP computations
    sample_idx = stratified_sample_indices(X_test, y_test, n=SAMPLE_SIZE, seed=42)
    X_s = X_test.loc[sample_idx]
    y_s = y_test.loc[sample_idx]
    ypred_s = pd.Series(y_pred, index=X_test.index).loc[sample_idx]

    # Assign category labels
    cat = pd.Series(index=sample_idx, dtype="object")
    cat[(y_s == 1) & (ypred_s == 1)] = "TP"
    cat[(y_s == 0) & (ypred_s == 1)] = "FP"
    cat[(y_s == 1) & (ypred_s == 0)] = "FN"
    cat[(y_s == 0) & (ypred_s == 0)] = "TN"

    print("Sample category counts:")
    print(cat.value_counts())

    # Compute SHAP on sampled subset
    explainer = shap.TreeExplainer(model)
    sv = compute_shap_attack_class(explainer, X_s)

    # Mean absolute SHAP per feature overall + by category
    results = []

    def mean_abs_for_index(ix):
        sv_ix = sv[[X_s.index.get_loc(i) for i in ix], :]
        return np.mean(np.abs(sv_ix), axis=0)

    # Overall
    overall = np.mean(np.abs(sv), axis=0)
    results.append(pd.DataFrame({
        "feature": feature_names,
        "category": "ALL",
        "mean_abs_shap": overall
    }))

    # Per category
    for c in ["TP", "FP", "FN", "TN"]:
        ix = cat[cat == c].index
        if len(ix) == 0:
            continue
        m = mean_abs_for_index(ix)
        results.append(pd.DataFrame({
            "feature": feature_names,
            "category": c,
            "mean_abs_shap": m
        }))

    out = pd.concat(results, ignore_index=True)

    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
    out.to_csv(OUT_TABLE, index=False)
    print(f"Saved SHAP error comparison table -> {OUT_TABLE}")

    # Top-K tables per category
    top_rows = []
    top_sets = {}
    for c in ["ALL", "TP", "FP", "FN", "TN"]:
        sub = out[out["category"] == c].set_index("feature")["mean_abs_shap"]
        if sub.empty:
            continue
        top = topk_features(sub, TOPK)
        top_sets[c] = list(top.index)
        for rank, (feat, val) in enumerate(top.items(), start=1):
            top_rows.append({"category": c, "rank": rank, "feature": feat, "mean_abs_shap": float(val)})

    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(OUT_TOPK, index=False)
    print(f"Saved Top-{TOPK} per category -> {OUT_TOPK}")

    # Overlap (Jaccard) between Top-K sets
    pairs = []
    cats = [c for c in ["TP", "FP", "FN", "TN"] if c in top_sets]
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            c1, c2 = cats[i], cats[j]
            pairs.append({
                "cat1": c1,
                "cat2": c2,
                "topk_jaccard": jaccard(top_sets[c1], top_sets[c2]),
                "shared_features": ", ".join(sorted(set(top_sets[c1]) & set(top_sets[c2])))
            })

    overlap_df = pd.DataFrame(pairs)
    overlap_df.to_csv(OUT_OVERLAP, index=False)
    print(f"Saved Top-{TOPK} overlap table -> {OUT_OVERLAP}")

    # Plots: Top features by category
    os.makedirs(FIG_DIR, exist_ok=True)

    for c in ["TP", "FP", "FN", "TN"]:
        sub = out[out["category"] == c].set_index("feature")["mean_abs_shap"]
        if sub.empty:
            continue
        top_plot = sub.sort_values(ascending=False).head(TOPPLOT).sort_values()
        plt.figure()
        plt.barh(top_plot.index, top_plot.values)
        plt.title(f"Top {TOPPLOT} mean(|SHAP|) features - {c}")
        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, f"shap_top{TOPPLOT}_{c}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved plot -> {fig_path}")

    # Optional: quick sanity note for your write-up
    print("Done.")

if __name__ == "__main__":
    main()