import os
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join("data", "processed", "cic_ids2017_clean.parquet")

MODEL_OUT = os.path.join("outputs", "models", "rf_binary.joblib")
REPORT_OUT = os.path.join("outputs", "tables", "binary_report.txt")
CM_OUT = os.path.join("outputs", "figures", "confusion_matrix_binary.png")
ROC_OUT = os.path.join("outputs", "figures", "roc_curve_binary.png")
PR_OUT = os.path.join("outputs", "figures", "precision_recall_curve_binary.png")
TRAIN_TEST_ROC_OUT = os.path.join("outputs", "figures", "roc_curve_train_vs_test.png")
AUC_RUNS_OUT = os.path.join("outputs", "tables", "roc_auc_repeated_runs.csv")
CV_OUT = os.path.join("outputs", "tables", "cross_validation_scores.csv")

# Fast settings for deadline-sensitive validation
N_RUNS = 15
REPEAT_SAMPLE_SIZE = 30000
CV_FOLDS = 3
TRAIN_SAMPLE_SIZE = 120000


def safe_positive_rate(y: pd.Series) -> float:
    return float(np.mean(y.astype(int)))


def build_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=60,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=1,   # avoid nested parallel slowdown
        random_state=random_state,
    )


def save_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Binary)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Benign", "Attack"])
    plt.yticks([0, 1], ["Benign", "Attack"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_roc_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    out_path: str,
    title: str = "ROC Curve (Binary)",
) -> tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.6f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return fpr, tpr, auc_score


def save_precision_recall_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    out_path: str,
    title: str = "Precision-Recall Curve (Binary)",
) -> tuple[np.ndarray, np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    baseline = safe_positive_rate(y_true)

    plt.figure()
    plt.plot(recall, precision, label=f"PR Curve (AP = {ap_score:.6f})")
    plt.hlines(
        y=baseline,
        xmin=0,
        xmax=1,
        linestyle="--",
        label=f"Baseline = {baseline:.6f}",
    )
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return precision, recall, ap_score


def save_train_test_roc(
    y_train: pd.Series,
    y_train_prob: np.ndarray,
    y_test: pd.Series,
    y_test_prob: np.ndarray,
    out_path: str,
) -> tuple[float, float]:
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)

    auc_train = roc_auc_score(y_train, y_train_prob)
    auc_test = roc_auc_score(y_test, y_test_prob)

    plt.figure()
    plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {auc_train:.6f})")
    plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {auc_test:.6f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.title("Train vs Test ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return auc_train, auc_test


def summarize_split_performance(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def main() -> None:
    for path in [
        MODEL_OUT,
        REPORT_OUT,
        CM_OUT,
        ROC_OUT,
        PR_OUT,
        TRAIN_TEST_ROC_OUT,
        AUC_RUNS_OUT,
        CV_OUT,
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.read_parquet(DATA)
    print(f"Loaded dataset: {df.shape}")

    y = df["label_binary"].astype(int)
    X = df.drop(columns=["label_binary", "label_original"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    X_train_fit, _, y_train_fit, _ = train_test_split(
        X_train,
        y_train,
        train_size=TRAIN_SAMPLE_SIZE,
        stratify=y_train,
        random_state=42,
    )
    print(f"Train subset used for fitting: {X_train_fit.shape}")

    clf = build_model(random_state=42)
    clf.fit(X_train_fit, y_train_fit)

    y_train_pred = clf.predict(X_train_fit)
    y_test_pred = clf.predict(X_test)

    y_train_prob = clf.predict_proba(X_train_fit)[:, 1]
    y_test_prob = clf.predict_proba(X_test)[:, 1]

    train_metrics = summarize_split_performance(y_train_fit, y_train_pred, y_train_prob)
    test_metrics = summarize_split_performance(y_test, y_test_pred, y_test_prob)

    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, digits=4)

    dump(clf, MODEL_OUT)

    save_confusion_matrix(cm, CM_OUT)

    _, _, test_auc = save_roc_curve(
        y_test,
        y_test_prob,
        ROC_OUT,
        title="ROC Curve (Binary, Test Set)",
    )
    _, _, test_ap = save_precision_recall_curve(
        y_test,
        y_test_prob,
        PR_OUT,
        title="Precision-Recall Curve (Binary, Test Set)",
    )

    train_auc, test_auc_again = save_train_test_roc(
        y_train_fit,
        y_train_prob,
        y_test,
        y_test_prob,
        TRAIN_TEST_ROC_OUT,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_model = build_model(random_state=42)

    cv_auc_scores = cross_val_score(
        cv_model,
        X_train_fit,
        y_train_fit,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
    )
    cv_acc_scores = cross_val_score(
        clone(cv_model),
        X_train_fit,
        y_train_fit,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )

    cv_df = pd.DataFrame(
        {
            "fold": list(range(1, CV_FOLDS + 1)),
            "roc_auc": cv_auc_scores,
            "accuracy": cv_acc_scores,
        }
    )
    cv_df.to_csv(CV_OUT, index=False)

    auc_scores = []
    X_rep, _, y_rep, _ = train_test_split(
        X,
        y,
        train_size=REPEAT_SAMPLE_SIZE,
        stratify=y,
        random_state=42,
    )

    for i in range(N_RUNS):
        print(f"Repeated run {i + 1}/{N_RUNS}...")
        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
            X_rep,
            y_rep,
            test_size=0.30,
            random_state=i,
            stratify=y_rep,
        )

        clf_i = RandomForestClassifier(
            n_estimators=40,
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=i,
        )

        clf_i.fit(X_train_i, y_train_i)
        y_prob_i = clf_i.predict_proba(X_test_i)[:, 1]
        auc_i = roc_auc_score(y_test_i, y_prob_i)
        auc_scores.append(auc_i)

    mean_auc = float(np.mean(auc_scores))
    std_auc = float(np.std(auc_scores))

    auc_df = pd.DataFrame(
        {
            "run": list(range(N_RUNS)),
            "roc_auc": auc_scores,
        }
    )
    auc_df.to_csv(AUC_RUNS_OUT, index=False)

    accuracy_gap = abs(train_metrics["accuracy"] - test_metrics["accuracy"])
    auc_gap = abs(train_metrics["roc_auc"] - test_metrics["roc_auc"])

    if accuracy_gap < 0.01 and auc_gap < 0.01:
        overfit_note = (
            "Train and test performance are very close, suggesting limited evidence "
            "of overfitting under this split."
        )
    else:
        overfit_note = (
            "There is a noticeable gap between train and test performance, which may "
            "suggest some degree of overfitting and should be interpreted with caution."
        )

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("Binary Intrusion Detection (CIC-IDS2017)\n")
        f.write("=================================================\n\n")

        f.write("Dataset Summary\n")
        f.write("------------------------------\n")
        f.write(f"Full dataset shape: {df.shape}\n")
        f.write(f"Train shape: {X_train.shape}\n")
        f.write(f"Test shape: {X_test.shape}\n")
        f.write(f"Train subset used for fitting: {X_train_fit.shape}\n")
        f.write(f"Attack prevalence (full): {safe_positive_rate(y):.6f}\n")
        f.write(f"Attack prevalence (train): {safe_positive_rate(y_train):.6f}\n")
        f.write(f"Attack prevalence (test): {safe_positive_rate(y_test):.6f}\n\n")

        f.write("Test Classification Report\n")
        f.write("------------------------------\n")
        f.write(report + "\n")

        f.write("Test Set Metrics\n")
        f.write("------------------------------\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.6f}\n")
        f.write(f"Precision: {test_metrics['precision']:.6f}\n")
        f.write(f"Recall: {test_metrics['recall']:.6f}\n")
        f.write(f"F1-score: {test_metrics['f1']:.6f}\n")
        f.write(f"ROC-AUC: {test_auc:.6f}\n")
        f.write(f"PR-AUC (Average Precision): {test_ap:.6f}\n\n")

        f.write("Confusion Matrix [ [TN, FP], [FN, TP] ]\n")
        f.write("------------------------------\n")
        f.write(f"{cm}\n\n")

        f.write("Train vs Test Generalisation Check\n")
        f.write("------------------------------\n")
        f.write(f"Train Accuracy: {train_metrics['accuracy']:.6f}\n")
        f.write(f"Test Accuracy: {test_metrics['accuracy']:.6f}\n")
        f.write(f"Train ROC-AUC: {train_auc:.6f}\n")
        f.write(f"Test ROC-AUC: {test_auc_again:.6f}\n")
        f.write(f"Absolute Accuracy Gap: {accuracy_gap:.6f}\n")
        f.write(f"Absolute ROC-AUC Gap: {auc_gap:.6f}\n")
        f.write(overfit_note + "\n\n")

        f.write(f"{CV_FOLDS}-Fold Cross-Validation on Training Subset\n")
        f.write("------------------------------\n")
        f.write(f"Accuracy scores: {np.array2string(cv_acc_scores, precision=6)}\n")
        f.write(f"Mean Accuracy: {cv_acc_scores.mean():.6f}\n")
        f.write(f"Std Accuracy: {cv_acc_scores.std():.6f}\n")
        f.write(f"ROC-AUC scores: {np.array2string(cv_auc_scores, precision=6)}\n")
        f.write(f"Mean ROC-AUC: {cv_auc_scores.mean():.6f}\n")
        f.write(f"Std ROC-AUC: {cv_auc_scores.std():.6f}\n\n")

        f.write(f"Repeated Evaluation ({N_RUNS} runs on subset)\n")
        f.write("------------------------------\n")
        f.write(f"Mean ROC-AUC: {mean_auc:.6f}\n")
        f.write(f"Std ROC-AUC: {std_auc:.6f}\n")

    print("\n=== Generalisation Check ===")
    print(f"Train Accuracy: {train_metrics['accuracy']:.6f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.6f}")
    print(f"Train ROC-AUC: {train_auc:.6f}")
    print(f"Test ROC-AUC: {test_auc_again:.6f}")
    print(f"Accuracy Gap: {accuracy_gap:.6f}")
    print(f"ROC-AUC Gap: {auc_gap:.6f}")
    print(overfit_note)

    print("\n=== Cross-Validation ===")
    print("CV Accuracy scores:", cv_acc_scores)
    print("Mean CV Accuracy:", cv_acc_scores.mean())
    print("CV ROC-AUC scores:", cv_auc_scores)
    print("Mean CV ROC-AUC:", cv_auc_scores.mean())

    print("\n=== Repeated Evaluation ===")
    print(f"Mean ROC-AUC ({N_RUNS} runs): {mean_auc:.6f}")
    print(f"Std ROC-AUC ({N_RUNS} runs): {std_auc:.6f}")

    print("\nSaved outputs:")
    print("Model:", MODEL_OUT)
    print("Report:", REPORT_OUT)
    print("Confusion Matrix:", CM_OUT)
    print("ROC Curve:", ROC_OUT)
    print("PR Curve:", PR_OUT)
    print("Train vs Test ROC:", TRAIN_TEST_ROC_OUT)
    print("Repeated AUC results:", AUC_RUNS_OUT)
    print("Cross-validation results:", CV_OUT)


if __name__ == "__main__":
    main()