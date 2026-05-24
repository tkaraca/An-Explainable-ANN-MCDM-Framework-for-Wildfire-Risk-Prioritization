from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.comparison import compare_methods, pairwise_spearman, topn_overlap
from src.data_utils import (
    LEAKY_COLUMNS,
    MISSING_CODE_COLUMNS,
    TARGET,
    add_training_only_features,
    apply_training_only_features,
    balanced_training_subset,
    basic_audit,
    build_feature_matrix,
    clean_base_features,
    evaluation_subset,
    load_raw_data,
    random_split,
    spatial_block_split,
    stratified_sample,
)
from src.mcdm import topsis_rank, topsis_rank_large, vikor_rank, vikor_rank_large
from src.modeling import (
    compute_critic_weights,
    compute_entropy_weights,
    compute_equal_weights,
    compute_permutation_weights,
    compute_shap_weights,
    evaluate_binary_model,
    fit_mlp,
    infer_feature_directions,
    predict_proba,
    predict_proba_batched,
)


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_all_grid_combos(text: str) -> list[tuple[str, str | None]]:
    combos: list[tuple[str, str | None]] = []
    for token in parse_csv_list(text):
        if token == "probability":
            combos.append(("probability", None))
        else:
            method, source = token.split(":", 1)
            combos.append((method.strip(), source.strip()))
    return combos


def score_col_name(method: str, weight_source: str | None = None) -> str:
    if method == "probability":
        return "score_probability_mlp"
    return f"score_{method}_{weight_source}"


def rank_col_name(method: str, weight_source: str | None = None) -> str:
    if method == "probability":
        return "rank_probability_mlp"
    return f"rank_{method}_{weight_source}"


def method_label(method: str, weight_source: str | None = None) -> str:
    if method == "probability":
        return "MLP_probability"
    return f"{method.upper()}__{weight_source}"


def save_audit(raw_df: pd.DataFrame, output_dir: Path) -> None:
    audit = basic_audit(raw_df)
    (output_dir / "prototype_data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    missing_rows = []
    for col in raw_df.columns:
        missing_rows.append(
            {
                "column": col,
                "dtype": str(raw_df[col].dtype),
                "missing": int(raw_df[col].isna().sum()),
                "coded_-9999": int((raw_df[col] == -9999).sum()) if col in MISSING_CODE_COLUMNS else 0,
                "n_unique": int(raw_df[col].nunique(dropna=False)),
            }
        )
    pd.DataFrame(missing_rows).sort_values(["missing", "coded_-9999"], ascending=False).to_csv(output_dir / "missing_profile.csv", index=False)

    leak_rows = []
    for col in LEAKY_COLUMNS:
        if col not in raw_df.columns:
            continue
        s = raw_df[col]
        row = {"column": col, "dtype": str(s.dtype), "missing": int(s.isna().sum()), "n_unique": int(s.nunique(dropna=False))}
        if pd.api.types.is_numeric_dtype(s):
            g0 = raw_df.loc[raw_df[TARGET] == 0, col]
            g1 = raw_df.loc[raw_df[TARGET] == 1, col]
            row["mean_target_0"] = float(g0.mean()) if g0.notna().any() else None
            row["mean_target_1"] = float(g1.mean()) if g1.notna().any() else None
        leak_rows.append(row)
    pd.DataFrame(leak_rows).to_csv(output_dir / "leakage_report.csv", index=False)


def save_weight_tables(weights_by_source: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    for source, df in weights_by_source.items():
        df.to_csv(weights_dir / f"weights_{source}.csv", index=False)
        combined.append(df.copy())
    combined_df = pd.concat(combined, ignore_index=True)
    combined_df.to_csv(output_dir / "feature_weights_all_sources.csv", index=False)
    pivot = combined_df.pivot_table(index="feature", columns="source", values="weight").reset_index()
    pivot.to_csv(output_dir / "feature_weights_pivot.csv", index=False)
    return combined_df


def save_schema(directions_df: pd.DataFrame, weights_by_source: dict[str, pd.DataFrame], output_dir: Path) -> None:
    base = directions_df.copy()
    for source, df in weights_by_source.items():
        renamed = df[["feature", "weight"]].rename(columns={"weight": f"weight_{source}"})
        base = base.merge(renamed, on="feature", how="left")
    base.to_csv(output_dir / "feature_schema_all_sources.csv", index=False)


def add_top_views(df: pd.DataFrame, score_columns: Iterable[str], output_dir: Path, top_n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for col in score_columns:
        top = df.sort_values(col, ascending=False).head(top_n).reset_index(drop=True)
        top.to_csv(output_dir / f"top_{top_n}_{col}.csv", index=False)


def aggregate_risk_blocks(scored: pd.DataFrame, score_columns: list[str], block_size: float = 1000.0) -> pd.DataFrame:
    if not {"Boylam", "Enlem"}.issubset(scored.columns):
        return pd.DataFrame()
    d = scored.copy()
    d["x_block"] = np.floor(d["Boylam"] / block_size).astype(int)
    d["y_block"] = np.floor(d["Enlem"] / block_size).astype(int)
    agg_map: dict[str, tuple[str, str]] = {
        "n_cells": (TARGET, "size"),
        "positive_cells": (TARGET, "sum"),
        "center_Boylam": ("Boylam", "mean"),
        "center_Enlem": ("Enlem", "mean"),
    }
    for col in score_columns:
        agg_map[f"mean_{col}"] = (col, "mean")
        agg_map[f"max_{col}"] = (col, "max")
    grouped = d.groupby(["x_block", "y_block"], as_index=False).agg(**agg_map)
    first_sort = f"mean_{score_columns[0]}" if score_columns else "n_cells"
    return grouped.sort_values(first_sort, ascending=False).reset_index(drop=True)


def run_mcdm_on_test(X_test: pd.DataFrame, weights_df: pd.DataFrame, directions_df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "topsis":
        return topsis_rank(X_test, weights_df, directions_df)
    if method == "vikor":
        return vikor_rank(X_test, weights_df, directions_df)
    raise ValueError(f"Bilinmeyen MCDM yöntemi: {method}")


def run_mcdm_on_all_grid(X_all: pd.DataFrame, weights_df: pd.DataFrame, directions_df: pd.DataFrame, method: str, batch_size: int) -> pd.DataFrame:
    if method == "topsis":
        return topsis_rank_large(X_all, weights_df, directions_df, batch_size=batch_size)
    if method == "vikor":
        return vikor_rank_large(X_all, weights_df, directions_df, batch_size=batch_size)
    raise ValueError(f"Bilinmeyen MCDM yöntemi: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Muğla orman yangını verisi için karşılaştırmalı YSA + MCDM pipeline")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="outputs_compare")
    parser.add_argument("--split", choices=["random", "spatial"], default="spatial")
    parser.add_argument("--neg-ratio", type=int, default=4)
    parser.add_argument("--eval-neg-ratio", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--include-coords", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--mlp-max-iter", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=250000)
    parser.add_argument("--skip-all-grid", action="store_true")
    parser.add_argument("--aggregate-block-size", type=float, default=1000.0)
    parser.add_argument("--weight-sources", default="pi,shap,equal,entropy,critic")
    parser.add_argument("--ranking-methods", default="probability,topsis,vikor")
    parser.add_argument("--all-grid-combos", default="probability,topsis:pi,vikor:pi,topsis:shap,vikor:shap")
    parser.add_argument("--all-grid-all-methods", action="store_true")
    parser.add_argument("--shap-background-size", type=int, default=50)
    parser.add_argument("--shap-explain-size", type=int, default=200)
    parser.add_argument("--shap-nsamples", default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings_list: list[str] = []

    print("[1/10] Ham veri okunuyor...")
    raw_df = load_raw_data(args.csv)
    raw_df = stratified_sample(raw_df, args.max_rows, random_state=42)
    save_audit(raw_df, output_dir)

    print("[2/10] Temizlik ve leakage kontrolü uygulanıyor...")
    cleaned = clean_base_features(raw_df)

    print(f"[3/10] {args.split} split hazırlanıyor...")
    split = random_split(cleaned) if args.split == "random" else spatial_block_split(cleaned)
    train_df, others, mapping, global_rate = add_training_only_features(split.train, [split.val, split.test])
    val_df, test_df = others

    X_train_full, y_train_full, _ = build_feature_matrix(train_df, include_coords=args.include_coords)
    X_val_full, y_val_full, meta_val = build_feature_matrix(val_df, include_coords=args.include_coords)
    X_test_full, y_test_full, meta_test = build_feature_matrix(test_df, include_coords=args.include_coords)

    X_train, y_train = balanced_training_subset(X_train_full, y_train_full, neg_ratio=args.neg_ratio)
    X_val, y_val, meta_val = evaluation_subset(X_val_full, y_val_full, meta_val, neg_ratio=args.eval_neg_ratio)
    X_test, y_test, meta_test = evaluation_subset(X_test_full, y_test_full, meta_test, neg_ratio=args.eval_neg_ratio)

    pd.DataFrame(
        [
            {"subset": "train_full", "rows": int(len(X_train_full))},
            {"subset": "val_full", "rows": int(len(X_val_full))},
            {"subset": "test_full", "rows": int(len(X_test_full))},
            {"subset": "balanced_train_used_for_mlp", "rows": int(len(X_train))},
            {"subset": "val_used_for_eval", "rows": int(len(X_val))},
            {"subset": "test_used_for_eval", "rows": int(len(X_test))},
        ]
    ).to_csv(output_dir / "split_summary.csv", index=False)

    print("[4/10] MLP eğitiliyor...")
    bundle = fit_mlp(X_train, y_train, max_iter=args.mlp_max_iter)

    print("[5/10] Model metrikleri hesaplanıyor...")
    metrics = evaluate_binary_model(bundle, X_val, y_val, X_test, y_test)
    metrics["note"] = "Test karşılaştırmaları örneklenmiş evaluation set üzerinde hesaplanır. Tam grid için ayrı skorlar üretilir."
    metrics["mlp_max_iter"] = args.mlp_max_iter
    (output_dir / "prototype_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[6/10] Feature direction ve ağırlık kaynakları çıkarılıyor...")
    directions_df = infer_feature_directions(X_train_full, y_train_full)
    directions_df.to_csv(output_dir / "feature_directions.csv", index=False)

    requested_weight_sources = [s for s in parse_csv_list(args.weight_sources) if s != "probability"]
    weights_by_source: dict[str, pd.DataFrame] = {}

    for source in requested_weight_sources:
        print(f"   - ağırlık kaynağı hazırlanıyor: {source}")
        try:
            if source == "pi":
                weights_by_source[source] = compute_permutation_weights(bundle, X_val, y_val)
            elif source == "shap":
                nsamples = args.shap_nsamples
                if nsamples != "auto":
                    nsamples = int(nsamples)
                weights_by_source[source] = compute_shap_weights(
                    bundle,
                    X_val,
                    background_size=args.shap_background_size,
                    explain_size=args.shap_explain_size,
                    nsamples=nsamples,
                )
            elif source == "equal":
                weights_by_source[source] = compute_equal_weights(bundle.feature_names)
            elif source == "entropy":
                weights_by_source[source] = compute_entropy_weights(X_train_full[bundle.feature_names], directions_df)
            elif source == "critic":
                weights_by_source[source] = compute_critic_weights(X_train_full[bundle.feature_names], directions_df)
            else:
                warnings_list.append(f"Bilinmeyen weight source atlandı: {source}")
        except Exception as exc:  # pragma: no cover
            warnings_list.append(f"Ağırlık kaynağı üretilemedi ({source}): {exc}")

    if not weights_by_source:
        raise RuntimeError("Hiçbir ağırlık kaynağı üretilemedi. weight-sources ayarını kontrol edin.")

    save_weight_tables(weights_by_source, output_dir)
    save_schema(directions_df, weights_by_source, output_dir)

    print("[7/10] Test örneklemi üzerinde karşılaştırmalı sıralamalar üretiliyor...")
    test_scores: dict[str, np.ndarray] = {}
    test_ranked = meta_test.copy()
    test_ranked[TARGET] = y_test
    prob_test = predict_proba(bundle, X_test)
    test_ranked[score_col_name("probability")] = prob_test
    test_ranked[rank_col_name("probability")] = pd.Series(prob_test).rank(ascending=False, method="first").astype(int).to_numpy()
    test_scores[method_label("probability")] = prob_test

    requested_rank_methods = parse_csv_list(args.ranking_methods)
    for method in requested_rank_methods:
        if method == "probability":
            continue
        for source, weights_df in weights_by_source.items():
            result = run_mcdm_on_test(X_test, weights_df, directions_df, method)
            score_col = score_col_name(method, source)
            rank_col = rank_col_name(method, source)
            test_ranked[score_col] = result.loc[test_ranked.index, "score"].to_numpy()
            test_ranked[rank_col] = result.loc[test_ranked.index, "rank"].to_numpy()
            test_scores[method_label(method, source)] = test_ranked[score_col].to_numpy()

    test_ranked.to_csv(output_dir / "test_ranked_comparison.csv", index=False)
    score_columns_test = [c for c in test_ranked.columns if c.startswith("score_")]
    add_top_views(test_ranked, score_columns_test, output_dir / "test_views", args.top_n)

    comparison_test = compare_methods(y_test, test_scores)
    comparison_test.to_csv(output_dir / "method_comparison_test.csv", index=False)
    pairwise_spearman(test_ranked[score_columns_test]).to_csv(output_dir / "method_spearman_test.csv", index=False)
    topn_overlap(test_ranked[score_columns_test], top_n=min(args.top_n, len(test_ranked))).to_csv(output_dir / "method_topn_overlap_test.csv", index=False)

    selected_all_grid_combos = parse_all_grid_combos(args.all_grid_combos)
    if args.all_grid_all_methods:
        selected_all_grid_combos = [("probability", None)]
        for method in requested_rank_methods:
            if method == "probability":
                continue
            for source in weights_by_source.keys():
                selected_all_grid_combos.append((method, source))

    if not args.skip_all_grid:
        print("[8/10] Tüm geçerli grid hücreleri skorlaniyor...")
        all_df = apply_training_only_features(cleaned, mapping, global_rate)
        X_all, y_all, meta_all = build_feature_matrix(all_df, include_coords=args.include_coords)
        all_scored = meta_all.copy()
        all_scored[TARGET] = y_all

        prob_all = predict_proba_batched(bundle, X_all, batch_size=args.batch_size)
        all_scored[score_col_name("probability")] = prob_all
        all_scored[rank_col_name("probability")] = pd.Series(prob_all).rank(ascending=False, method="first").astype(int).to_numpy()

        for method, source in selected_all_grid_combos:
            if method == "probability":
                continue
            if source not in weights_by_source:
                warnings_list.append(f"All-grid kombinasyonu atlandı: {method}:{source} (weight source yok)")
                continue
            print(f"   - all-grid skorlanıyor: {method}:{source}")
            result = run_mcdm_on_all_grid(X_all, weights_by_source[source], directions_df, method, batch_size=args.batch_size)
            score_col = score_col_name(method, source)
            rank_col = rank_col_name(method, source)
            all_scored[score_col] = result.loc[all_scored.index, "score"].to_numpy()
            all_scored[rank_col] = result.loc[all_scored.index, "rank"].to_numpy()

        all_scored.to_csv(output_dir / "all_grid_scored.csv", index=False)
        score_columns_all = [c for c in all_scored.columns if c.startswith("score_")]
        add_top_views(all_scored, score_columns_all, output_dir / "all_grid_views", args.top_n)
        comparison_all = compare_methods(y_all, {col: all_scored[col].to_numpy() for col in score_columns_all})
        comparison_all.to_csv(output_dir / "method_comparison_all_grid.csv", index=False)
        aggregate_risk_blocks(all_scored, score_columns_all, block_size=args.aggregate_block_size).to_csv(
            output_dir / f"risk_blocks_{int(args.aggregate_block_size)}m.csv", index=False
        )
    else:
        print("[8/10] Tüm grid skorlama atlandı.")

    print("[9/10] Çalışma özeti ve uyarılar yazılıyor...")
    run_meta = {
        "weight_sources_requested": requested_weight_sources,
        "weight_sources_completed": list(weights_by_source.keys()),
        "ranking_methods_requested": requested_rank_methods,
        "all_grid_combos": ["probability" if m == "probability" else f"{m}:{s}" for m, s in selected_all_grid_combos],
        "warnings": warnings_list,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[10/10] İşlem tamamlandı.")
    print(f"Çıktı klasörü: {output_dir.resolve()}")
    if warnings_list:
        print("Not: bazı isteğe bağlı bileşenler uyarı verdi; run_metadata.json dosyasını kontrol edin.")


if __name__ == "__main__":
    main()
