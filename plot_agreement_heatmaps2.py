from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalize_method_name(name: str) -> str:
    """
    Standardize method names so that files with slightly different naming
    conventions can still be matched and plotted correctly.
    """
    if name is None:
        return ""

    s = str(name).strip()

    replacements = {
        "MLP probability": "MLP probability",
        "MLP_probability": "MLP probability",
        "mlp_probability": "MLP probability",
        "score_probability_mlp": "MLP probability",

        "TOPSIS__critic": "TOPSIS (CRITIC)",
        "TOPSIS (CRITIC)": "TOPSIS (CRITIC)",
        "score_topsis_critic": "TOPSIS (CRITIC)",

        "TOPSIS__entropy": "TOPSIS (Entropy)",
        "TOPSIS (Entropy)": "TOPSIS (Entropy)",
        "score_topsis_entropy": "TOPSIS (Entropy)",

        "TOPSIS__equal": "TOPSIS (Equal)",
        "TOPSIS (Equal)": "TOPSIS (Equal)",
        "score_topsis_equal": "TOPSIS (Equal)",

        "TOPSIS__pi": "TOPSIS (PI)",
        "TOPSIS (PI)": "TOPSIS (PI)",
        "score_topsis_pi": "TOPSIS (PI)",

        "TOPSIS__shap": "TOPSIS (SHAP)",
        "TOPSIS (SHAP)": "TOPSIS (SHAP)",
        "score_topsis_shap": "TOPSIS (SHAP)",

        "VIKOR__critic": "VIKOR (CRITIC)",
        "VIKOR (CRITIC)": "VIKOR (CRITIC)",
        "score_vikor_critic": "VIKOR (CRITIC)",

        "VIKOR__entropy": "VIKOR (Entropy)",
        "VIKOR (Entropy)": "VIKOR (Entropy)",
        "score_vikor_entropy": "VIKOR (Entropy)",

        "VIKOR__equal": "VIKOR (Equal)",
        "VIKOR (Equal)": "VIKOR (Equal)",
        "score_vikor_equal": "VIKOR (Equal)",

        "VIKOR__pi": "VIKOR (PI)",
        "VIKOR (PI)": "VIKOR (PI)",
        "score_vikor_pi": "VIKOR (PI)",

        "VIKOR__shap": "VIKOR (SHAP)",
        "VIKOR (SHAP)": "VIKOR (SHAP)",
        "score_vikor_shap": "VIKOR (SHAP)",
    }

    return replacements.get(s, s)


def display_label(name: str) -> str:
    """
    Convert long method labels into shorter two-line labels for cleaner axes.
    """
    mapping = {
        "MLP probability": "MLP",
        "TOPSIS (CRITIC)": "TOPSIS\n(CRITIC)",
        "TOPSIS (Entropy)": "TOPSIS\n(Entropy)",
        "TOPSIS (Equal)": "TOPSIS\n(Equal)",
        "TOPSIS (PI)": "TOPSIS\n(PI)",
        "TOPSIS (SHAP)": "TOPSIS\n(SHAP)",
        "VIKOR (CRITIC)": "VIKOR\n(CRITIC)",
        "VIKOR (Entropy)": "VIKOR\n(Entropy)",
        "VIKOR (Equal)": "VIKOR\n(Equal)",
        "VIKOR (PI)": "VIKOR\n(PI)",
        "VIKOR (SHAP)": "VIKOR\n(SHAP)",
    }
    return mapping.get(name, name.replace(" ", "\n", 1))


def read_pairwise_matrix(csv_path: Path, value_column: str):
    """
    Read a long-format pairwise comparison table and convert it into a square matrix.
    Expected columns include method_1 / method_2 or similar equivalents.
    """
    df = pd.read_csv(csv_path)

    # detect pair columns
    cols = list(df.columns)
    col1_candidates = ["method_1", "method_a", "row_method", "method_x", "left_method"]
    col2_candidates = ["method_2", "method_b", "col_method", "method_y", "right_method"]

    col1 = next((c for c in col1_candidates if c in cols), None)
    col2 = next((c for c in col2_candidates if c in cols), None)

    if col1 is None or col2 is None:
        raise ValueError(
            f"Could not detect method pair columns in {csv_path.name}. "
            f"Found columns: {cols}"
        )

    if value_column not in cols:
        raise ValueError(
            f"Column '{value_column}' not found in {csv_path.name}. "
            f"Found columns: {cols}"
        )

    df[col1] = df[col1].map(normalize_method_name)
    df[col2] = df[col2].map(normalize_method_name)

    methods = sorted(set(df[col1]).union(set(df[col2])))

    # Use custom order if all standard methods are present
    preferred_order = [
        "MLP probability",
        "TOPSIS (CRITIC)",
        "TOPSIS (Entropy)",
        "TOPSIS (Equal)",
        "TOPSIS (PI)",
        "TOPSIS (SHAP)",
        "VIKOR (CRITIC)",
        "VIKOR (Entropy)",
        "VIKOR (Equal)",
        "VIKOR (PI)",
        "VIKOR (SHAP)",
    ]

    ordered_methods = [m for m in preferred_order if m in methods]
    ordered_methods += [m for m in methods if m not in ordered_methods]

    n = len(ordered_methods)
    matrix = np.full((n, n), np.nan)

    idx = {m: i for i, m in enumerate(ordered_methods)}

    for _, row in df.iterrows():
        m1 = row[col1]
        m2 = row[col2]
        val = row[value_column]

        i = idx[m1]
        j = idx[m2]
        matrix[i, j] = val
        matrix[j, i] = val

    # diagonal = 1
    for i in range(n):
        matrix[i, i] = 1.0

    return ordered_methods, matrix


def find_overlap_value_column(df: pd.DataFrame):
    candidates = [
        "jaccard",
        "jaccard_overlap",
        "top1000_jaccard",
        "overlap_jaccard",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find Jaccard column. Found columns: {list(df.columns)}"
    )


def annotate_heatmap(ax, matrix, fontsize=8):
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if np.isnan(val):
                txt = ""
            else:
                txt = f"{val:.3f}"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=fontsize
            )


def main():
    parser = argparse.ArgumentParser(
        description="Plot agreement heatmaps (Spearman and Top-1000 Jaccard overlap)."
    )
    parser.add_argument(
        "--spearman-csv",
        type=str,
        required=True,
        help="Path to method_spearman_test.csv"
    )
    parser.add_argument(
        "--overlap-csv",
        type=str,
        required=True,
        help="Path to method_topn_overlap_test.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figure_agreement_heatmaps.png",
        help="Output figure path"
    )
    args = parser.parse_args()

    spearman_csv = Path(args.spearman_csv)
    overlap_csv = Path(args.overlap_csv)
    output = Path(args.output)

    if not spearman_csv.exists():
        raise FileNotFoundError(f"File not found: {spearman_csv}")
    if not overlap_csv.exists():
        raise FileNotFoundError(f"File not found: {overlap_csv}")

    # ---- Read Spearman matrix
    methods_s, spearman_matrix = read_pairwise_matrix(
        spearman_csv,
        value_column="spearman_rho"
    )

    # ---- Read overlap/Jaccard matrix
    overlap_df = pd.read_csv(overlap_csv)
    overlap_value_column = find_overlap_value_column(overlap_df)

    # Save temporarily with normalized names
    overlap_df = overlap_df.copy()
    col1 = next((c for c in ["method_1", "method_a", "row_method", "method_x", "left_method"] if c in overlap_df.columns), None)
    col2 = next((c for c in ["method_2", "method_b", "col_method", "method_y", "right_method"] if c in overlap_df.columns), None)

    if col1 is None or col2 is None:
        raise ValueError(
            f"Could not detect method pair columns in {overlap_csv.name}. "
            f"Found columns: {list(overlap_df.columns)}"
        )

    overlap_df[col1] = overlap_df[col1].map(normalize_method_name)
    overlap_df[col2] = overlap_df[col2].map(normalize_method_name)

    # Use same order as Spearman for consistency
    methods = methods_s
    idx = {m: i for i, m in enumerate(methods)}
    n = len(methods)

    jaccard_matrix = np.full((n, n), np.nan)

    for _, row in overlap_df.iterrows():
        m1 = row[col1]
        m2 = row[col2]
        if m1 in idx and m2 in idx:
            i = idx[m1]
            j = idx[m2]
            jaccard_matrix[i, j] = row[overlap_value_column]
            jaccard_matrix[j, i] = row[overlap_value_column]

    for i in range(n):
        jaccard_matrix[i, i] = 1.0

    # ---- Labels
    xlabels = [display_label(m) for m in methods]
    ylabels = [display_label(m) for m in methods]

    # ---- Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    im1 = axes[0].imshow(spearman_matrix, vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("Spearman rank correlation", fontsize=15)

    im2 = axes[1].imshow(jaccard_matrix, vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("Top-1000 Jaccard overlap", fontsize=15)

    for ax in axes:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))

        ax.set_xticklabels(
            xlabels,
            rotation=35,
            ha="right",
            rotation_mode="anchor",
            fontsize=9
        )
        ax.set_yticklabels(
            ylabels,
            fontsize=10
        )

    annotate_heatmap(axes[0], spearman_matrix, fontsize=8)
    annotate_heatmap(axes[1], jaccard_matrix, fontsize=8)

    fig.suptitle("Agreement and divergence among ranking strategies", fontsize=17)

    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Spearman rho", fontsize=12)

    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("Jaccard overlap", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, wspace=0.30)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {output.resolve()}")


if __name__ == "__main__":
    main()