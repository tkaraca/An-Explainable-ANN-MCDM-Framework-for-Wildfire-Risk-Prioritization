from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.data_utils import (
    COORDS,
    TARGET,
    add_training_only_features,
    clean_base_features,
    load_raw_data,
    random_split,
    spatial_block_split,
    stratified_sample,
)
from src.modeling import fit_mlp, evaluate_binary_model, predict_proba


DEFAULT_LEAKY_NUMERIC = ["brightness", "confidence", "frp", "scan", "track", "bright_t31"]
DEFAULT_LEAKY_PRESENCE = [
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "version",
    "bright_t31",
    "daynight",
    "type",
]


def make_sample_indices(y: np.ndarray, neg_ratio: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.arange(len(y))
    neg_sample_size = min(len(neg_idx), max(len(pos_idx), len(pos_idx) * max(1, neg_ratio)))
    neg_sample_idx = rng.choice(neg_idx, size=neg_sample_size, replace=False)
    idx = np.concatenate([pos_idx, neg_sample_idx])
    rng.shuffle(idx)
    return idx


def build_safe_and_leaky_feature_sets(raw_split: pd.DataFrame, mapping: Dict, global_rate: float | None) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    cleaned = clean_base_features(raw_split)
    if mapping is not None:
        cleaned["BitkiRiskSkoru"] = cleaned["Bitki_Turu"].map(mapping).fillna(global_rate)
    else:
        raise ValueError("Mapping/global_rate must be provided.")

    safe_drop = [TARGET, "Bitki_Turu"] + [c for c in COORDS if c in cleaned.columns]
    X_safe = cleaned.drop(columns=[c for c in safe_drop if c in cleaned.columns]).copy()
    y = cleaned[TARGET].astype(int).to_numpy()
    meta = raw_split[[c for c in COORDS if c in raw_split.columns]].reset_index(drop=True)

    extra = pd.DataFrame(index=raw_split.index)
    for col in DEFAULT_LEAKY_NUMERIC:
        if col in raw_split.columns:
            extra[f"leaky_{col}"] = pd.to_numeric(raw_split[col], errors="coerce")
    for col in DEFAULT_LEAKY_PRESENCE:
        if col in raw_split.columns:
            extra[f"has_{col}"] = raw_split[col].notna().astype(int)

    X_leaky = pd.concat([X_safe.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    return X_safe.reset_index(drop=True), y, X_leaky.reset_index(drop=True)



def subset_df(X: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    return X.iloc[idx].reset_index(drop=True)



def run_single_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    max_iter: int = 150,
    random_state: int = 42,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    bundle = fit_mlp(X_train, y_train, max_iter=max_iter, random_state=random_state)
    metrics = evaluate_binary_model(bundle, X_val, y_val, X_test, y_test)
    val_probs = predict_proba(bundle, X_val)
    test_probs = predict_proba(bundle, X_test)
    metrics["method"] = name
    metrics["n_train"] = int(len(X_train))
    metrics["n_val"] = int(len(X_val))
    metrics["n_test"] = int(len(X_test))
    metrics["n_features"] = int(X_train.shape[1])
    return metrics, val_probs, test_probs



def save_curve_plot(y_test: np.ndarray, score_map: Dict[str, np.ndarray], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes
    for label, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = roc_auc_score(y_test, scores)
        ax1.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(fontsize=8)

    prevalence = float(np.mean(y_test))
    for label, scores in score_map.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = average_precision_score(y_test, scores)
        ax2.plot(recall, precision, label=f"{label} (AP={pr_auc:.4f})")
    ax2.axhline(prevalence, linestyle="--", linewidth=1)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision–Recall Curve")
    ax2.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-control experiments for Muğla wildfire ANN pipeline")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="outputs_leakage_controls")
    parser.add_argument("--split", choices=["random", "spatial"], default="spatial")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--neg-ratio", type=int, default=4)
    parser.add_argument("--eval-neg-ratio", type=int, default=10)
    parser.add_argument("--mlp-max-iter", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data(args.csv)
    raw_df = stratified_sample(raw_df, args.max_rows, random_state=args.random_state)
    split = random_split(raw_df, random_state=args.random_state) if args.split == "random" else spatial_block_split(raw_df, random_state=args.random_state)

    train_clean_base = clean_base_features(split.train)
    val_clean_base = clean_base_features(split.val)
    test_clean_base = clean_base_features(split.test)
    train_clean, others, mapping, global_rate = add_training_only_features(train_clean_base, [val_clean_base, test_clean_base])
    val_clean, test_clean = others

    X_train_safe_full, y_train_full, X_train_leaky_full = build_safe_and_leaky_feature_sets(split.train, mapping, global_rate)
    X_val_safe_full, y_val_full, X_val_leaky_full = build_safe_and_leaky_feature_sets(split.val, mapping, global_rate)
    X_test_safe_full, y_test_full, X_test_leaky_full = build_safe_and_leaky_feature_sets(split.test, mapping, global_rate)

    train_idx = make_sample_indices(y_train_full, args.neg_ratio, args.random_state)
    val_idx = make_sample_indices(y_val_full, args.eval_neg_ratio, args.random_state + 1)
    test_idx = make_sample_indices(y_test_full, args.eval_neg_ratio, args.random_state + 2)

    X_train_safe = subset_df(X_train_safe_full, train_idx)
    y_train = y_train_full[train_idx]
    X_val_safe = subset_df(X_val_safe_full, val_idx)
    y_val = y_val_full[val_idx]
    X_test_safe = subset_df(X_test_safe_full, test_idx)
    y_test = y_test_full[test_idx]

    X_train_leaky = subset_df(X_train_leaky_full, train_idx)
    X_val_leaky = subset_df(X_val_leaky_full, val_idx)
    X_test_leaky = subset_df(X_test_leaky_full, test_idx)

    X_train_coords = split.train[COORDS].iloc[train_idx].reset_index(drop=True)
    X_val_coords = split.val[COORDS].iloc[val_idx].reset_index(drop=True)
    X_test_coords = split.test[COORDS].iloc[test_idx].reset_index(drop=True)

    results = []
    score_map_test: Dict[str, np.ndarray] = {}

    baseline_metrics, _, baseline_test_probs = run_single_model(
        "baseline_ann",
        X_train_safe, y_train, X_val_safe, y_val, X_test_safe, y_test,
        max_iter=args.mlp_max_iter, random_state=args.random_state,
    )
    results.append(baseline_metrics)
    score_map_test["Baseline ANN"] = baseline_test_probs

    rng = np.random.default_rng(args.random_state)
    y_train_perm = y_train.copy()
    rng.shuffle(y_train_perm)
    labelperm_metrics, _, labelperm_test_probs = run_single_model(
        "label_permutation",
        X_train_safe, y_train_perm, X_val_safe, y_val, X_test_safe, y_test,
        max_iter=args.mlp_max_iter, random_state=args.random_state,
    )
    results.append(labelperm_metrics)
    score_map_test["Label permutation"] = labelperm_test_probs

    coord_metrics, _, coord_test_probs = run_single_model(
        "coordinate_only",
        X_train_coords, y_train, X_val_coords, y_val, X_test_coords, y_test,
        max_iter=args.mlp_max_iter, random_state=args.random_state,
    )
    results.append(coord_metrics)
    score_map_test["Coordinates only"] = coord_test_probs

    leaky_metrics, _, leaky_test_probs = run_single_model(
        "deliberately_leaky",
        X_train_leaky, y_train, X_val_leaky, y_val, X_test_leaky, y_test,
        max_iter=args.mlp_max_iter, random_state=args.random_state,
    )
    results.append(leaky_metrics)
    score_map_test["Deliberately leaky"] = leaky_test_probs

    summary = pd.DataFrame(results)
    summary = summary[[
        "method", "n_features", "n_train", "n_val", "n_test",
        "val_roc_auc", "val_pr_auc", "test_roc_auc", "test_pr_auc", "test_f1", "decision_threshold"
    ]].sort_values("test_pr_auc", ascending=False)
    summary.to_csv(out_dir / "leakage_controls_summary.csv", index=False)
    (out_dir / "leakage_controls_summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")

    pd.DataFrame({"feature": X_train_safe.columns}).to_csv(out_dir / "baseline_safe_features.csv", index=False)
    pd.DataFrame({"feature": X_train_leaky.columns}).to_csv(out_dir / "deliberately_leaky_features.csv", index=False)

    save_curve_plot(y_test, score_map_test, out_dir / "leakage_controls_roc_pr.png", title=f"Leakage control experiments ({args.split} split)")

    notes = {
        "split": args.split,
        "max_rows": args.max_rows,
        "train_full_rows": int(len(split.train)),
        "val_full_rows": int(len(split.val)),
        "test_full_rows": int(len(split.test)),
        "train_used_rows": int(len(X_train_safe)),
        "val_used_rows": int(len(X_val_safe)),
        "test_used_rows": int(len(X_test_safe)),
        "test_positive_rate": float(np.mean(y_test)),
        "leaky_numeric_features_added": [f"leaky_{c}" for c in DEFAULT_LEAKY_NUMERIC if c in split.train.columns],
        "leaky_presence_indicators_added": [f"has_{c}" for c in DEFAULT_LEAKY_PRESENCE if c in split.train.columns],
    }
    (out_dir / "leakage_controls_notes.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Leakage control outputs written to:", out_dir.resolve())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
