import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

LABELS = {
    'baseline_ann': 'Baseline ANN\n(safe features)',
    'label_permutation': 'Label permutation',
    'coordinate_only': 'Coordinate-only\nbaseline',
    'deliberately_leaky': 'Deliberately leaky\nspecification',
}
ORDER = ['baseline_ann', 'label_permutation', 'coordinate_only', 'deliberately_leaky']


def main():
    parser = argparse.ArgumentParser(description='Plot leakage-control comparison metrics.')
    parser.add_argument('--csv', required=True, help='Path to leakage_controls_summary.csv')
    parser.add_argument('--output', default=None, help='Output PNG path. Defaults next to CSV.')
    parser.add_argument('--title', default='Leakage-control and negative-control comparisons', help='Figure title')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.output) if args.output else csv_path.with_name('leakage_controls_barplot.png')

    df = pd.read_csv(csv_path)
    df = df[df['method'].isin(ORDER)].copy()
    df['order'] = df['method'].apply(lambda x: ORDER.index(x))
    df = df.sort_values('order')
    labels = [LABELS.get(m, m) for m in df['method']]

    metrics = ['test_roc_auc', 'test_pr_auc', 'test_f1']
    metric_names = ['ROC-AUC', 'PR-AUC', 'F1-score']

    x = list(range(len(df)))
    width = 0.23

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for idx, (metric, mname) in enumerate(zip(metrics, metric_names)):
        xpos = [i + (idx - 1) * width for i in x]
        vals = df[metric].tolist()
        bars = ax.bar(xpos, vals, width=width, label=mname)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score')
    ax.set_title(args.title)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3)

    note = (
        'Safe features exclude predefined leakage-prone fire-detection/post-event variables. '
        'The deliberately leaky specification reintroduces these variables as a counterfactual control.'
    )
    fig.text(0.01, 0.01, note, ha='left', va='bottom', fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved figure to: {out_path}')


if __name__ == '__main__':
    main()
