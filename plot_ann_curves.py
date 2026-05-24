from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def infer_label_col(df: pd.DataFrame) -> str:
    candidates = ["YANGIN_DURUMU", "target", "y", "label"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Label column not found. Expected one of: " + ", ".join(candidates)
    )


def infer_score_col(df: pd.DataFrame) -> str:
    candidates = ["score_probability_mlp", "probability", "mlp_probability", "score"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "ANN score column not found. Expected one of: " + ", ".join(candidates)
    )


def safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        return float(x)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ROC and Precision-Recall curves for the ANN baseline (MLP_probability)."
    )
    parser.add_argument(
        "--csv",
        default="outputs_full_compare/test_ranked_comparison.csv",
        help="Path to test_ranked_comparison.csv",
    )
    parser.add_argument(
        "--metrics-json",
        default="outputs_full_compare/prototype_metrics.json",
        help="Optional path to prototype_metrics.json for threshold annotation",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Label column name. Default: auto-detect (usually YANGIN_DURUMU)",
    )
    parser.add_argument(
        "--score-col",
        default=None,
        help="Score column name. Default: auto-detect (usually score_probability_mlp)",
    )
    parser.add_argument(
        "--title-prefix",
        default="ANN Baseline (MLP)",
        help="Prefix used in figure titles",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save outputs. Default: same folder as the input CSV",
    )
    parser.add_argument(
        "--prefix",
        default="ann_baseline",
        help="Filename prefix for output figures",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    label_col = args.label_col or infer_label_col(df)
    score_col = args.score_col or infer_score_col(df)

    work = df[[label_col, score_col]].copy()
    work = work.dropna()
    y_true = work[label_col].astype(int).to_numpy()
    y_score = work[score_col].astype(float).to_numpy()

    if len(np.unique(y_true)) != 2:
        raise ValueError("Label column must contain both classes (0 and 1).")

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    prevalence = float(np.mean(y_true))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    metrics_info: dict[str, Any] = {}
    metrics_path = Path(args.metrics_json)
    threshold = None
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics_info = json.load(f)
        threshold = safe_float(metrics_info.get("threshold"))

    if threshold is None:
        pred = (y_score >= 0.5).astype(int)
        threshold = 0.5
        f1_at_threshold = f1_score(y_true, pred)
    else:
        pred = (y_score >= threshold).astype(int)
        f1_at_threshold = f1_score(y_true, pred)

    # ROC figure
    fig1 = plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ANN ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{args.title_prefix} — ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_png = output_dir / f"{args.prefix}_roc_curve.png"
    fig1.savefig(roc_png, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # PR figure
    fig2 = plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, linewidth=2, label=f"ANN PR (AP = {pr_auc:.4f})")
    plt.axhline(prevalence, linestyle="--", linewidth=1, label=f"Prevalence = {prevalence:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{args.title_prefix} — Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_png = output_dir / f"{args.prefix}_pr_curve.png"
    fig2.savefig(pr_png, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # Combined figure
    fig3 = plt.figure(figsize=(12, 5))
    ax1 = fig3.add_subplot(1, 2, 1)
    ax1.plot(fpr, tpr, linewidth=2, label=f"ANN ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random classifier")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")

    ax2 = fig3.add_subplot(1, 2, 2)
    ax2.plot(recall, precision, linewidth=2, label=f"ANN PR (AP = {pr_auc:.4f})")
    ax2.axhline(prevalence, linestyle="--", linewidth=1, label=f"Prevalence = {prevalence:.4f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="lower left")

    fig3.suptitle(f"{args.title_prefix} — Baseline Evaluation", y=1.02)
    fig3.tight_layout()
    combined_png = output_dir / f"{args.prefix}_roc_pr_combined.png"
    fig3.savefig(combined_png, dpi=300, bbox_inches="tight")
    plt.close(fig3)

    summary = {
        "csv": str(csv_path),
        "label_col": label_col,
        "score_col": score_col,
        "n_samples": int(len(work)),
        "n_positive": int(np.sum(y_true)),
        "positive_rate": prevalence,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold_used_for_f1": float(threshold),
        "f1_at_threshold": float(f1_at_threshold),
        "metrics_json_found": metrics_path.exists(),
        "roc_curve_png": str(roc_png),
        "pr_curve_png": str(pr_png),
        "combined_png": str(combined_png),
    }
    summary_json = output_dir / f"{args.prefix}_curve_metrics.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("ANN evaluation plots created successfully.")
    print(f"ROC curve: {roc_png}")
    print(f"PR curve: {pr_png}")
    print(f"Combined figure: {combined_png}")
    print(f"Summary JSON: {summary_json}")
    print(f"ROC-AUC = {roc_auc:.4f}")
    print(f"PR-AUC  = {pr_auc:.4f}")
    print(f"F1 @ threshold {threshold:.4f} = {f1_at_threshold:.4f}")


if __name__ == "__main__":
    main()
