from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalize_method_name(name: str) -> str:
    mapping = {
        'score_probability_mlp': 'MLP\nprobability',
        'score_topsis_pi': 'TOPSIS\n(PI)',
        'score_vikor_pi': 'VIKOR\n(PI)',
        'score_topsis_shap': 'TOPSIS\n(SHAP)',
        'score_vikor_shap': 'VIKOR\n(SHAP)',
        'score_topsis_equal': 'TOPSIS\n(Equal)',
        'score_vikor_equal': 'VIKOR\n(Equal)',
        'score_topsis_entropy': 'TOPSIS\n(Entropy)',
        'score_vikor_entropy': 'VIKOR\n(Entropy)',
        'score_topsis_critic': 'TOPSIS\n(CRITIC)',
        'score_vikor_critic': 'VIKOR\n(CRITIC)',
    }
    return mapping.get(name, name)


def build_symmetric_matrix(df, value_col, methods):
    M = pd.DataFrame(np.nan, index=methods, columns=methods)
    np.fill_diagonal(M.values, 1.0)
    for _, row in df.iterrows():
        a = row['method_a']
        b = row['method_b']
        v = row[value_col]
        if a in methods and b in methods:
            M.loc[a, b] = v
            M.loc[b, a] = v
    return M


def annotate_heatmap(ax, matrix, fmt='{:.3f}', fontsize=8):
    vals = matrix.values
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, fmt.format(vals[i, j]), ha='center', va='center', fontsize=fontsize)


def main():
    parser = argparse.ArgumentParser(description='Plot Spearman and top-1000 Jaccard heatmaps for ranking agreement.')
    parser.add_argument('--spearman-csv', type=str, default='outputs_full_compare/method_spearman_test.csv')
    parser.add_argument('--overlap-csv', type=str, default='outputs_full_compare/method_topn_overlap_test.csv')
    parser.add_argument('--output', type=str, default='figure_agreement_heatmaps.png')
    parser.add_argument('--main-only', action='store_true', help='Use only main methods: MLP, PI-TOPSIS/VIKOR, SHAP-TOPSIS/VIKOR')
    args = parser.parse_args()

    sp = pd.read_csv(args.spearman_csv)
    ov = pd.read_csv(args.overlap_csv)
    ov = ov[ov['top_n'] == 1000].copy()

    if args.main_only:
        methods = [
            'score_probability_mlp',
            'score_topsis_pi',
            'score_vikor_pi',
            'score_topsis_shap',
            'score_vikor_shap',
        ]
    else:
        methods = sorted(set(sp['method_a']).union(sp['method_b']))

    spM = build_symmetric_matrix(sp, 'spearman_rho', methods)
    ovM = build_symmetric_matrix(ov, 'jaccard', methods)

    labels = [normalize_method_name(m) for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    im1 = axes[0].imshow(spM.values, vmin=0, vmax=1)
    axes[0].set_title('Spearman rank correlation')
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_xticklabels(labels, rotation=35, ha='right')
    axes[0].set_yticklabels(labels)
    annotate_heatmap(axes[0], spM, fmt='{:.3f}', fontsize=7)
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Spearman rho')

    im2 = axes[1].imshow(ovM.values, vmin=0, vmax=1)
    axes[1].set_title('Top-1000 Jaccard overlap')
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_yticks(range(len(methods)))
    axes[1].set_xticklabels(labels, rotation=35, ha='right')
    axes[1].set_yticklabels(labels)
    annotate_heatmap(axes[1], ovM, fmt='{:.3f}', fontsize=7)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Jaccard overlap')

    fig.suptitle('Agreement and divergence among ranking strategies', y=0.98)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved figure to: {Path(args.output).resolve()}')


if __name__ == '__main__':
    main()
