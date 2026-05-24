from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def aggregate_to_blocks(df: pd.DataFrame, block_size: int = 1000) -> pd.DataFrame:
    out = df.copy()
    out['x_block'] = np.floor(out['Boylam'] / block_size).astype(int)
    out['y_block'] = np.floor(out['Enlem'] / block_size).astype(int)
    grouped = (
        out.groupby(['x_block', 'y_block'], as_index=False)
        .agg(
            Boylam=('Boylam', 'mean'),
            Enlem=('Enlem', 'mean'),
            mean_score_probability_mlp=('score_probability_mlp', 'mean'),
            mean_score_topsis_pi=('score_topsis_pi', 'mean'),
        )
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot full-grid wildfire priority maps.')
    parser.add_argument(
        '--csv',
        default='outputs_full_compare/all_grid_scored.csv',
        help='Path to all_grid_scored.csv (default: outputs_full_compare/all_grid_scored.csv)',
    )
    parser.add_argument(
        '--output',
        default='outputs_full_compare/figure_full_grid_maps.png',
        help='Output figure path (default: outputs_full_compare/figure_full_grid_maps.png)',
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=1000,
        help='Aggregation block size in projected map units (default: 1000)',
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    csv_path = (base / args.csv).resolve() if not Path(args.csv).is_absolute() else Path(args.csv)
    out_path = (base / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    usecols = ['Boylam', 'Enlem', 'score_probability_mlp', 'score_topsis_pi']
    df = pd.read_csv(csv_path, usecols=usecols)
    agg = aggregate_to_blocks(df, block_size=args.block_size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    cmap = 'inferno'
    vmin, vmax = 0.0, 1.0

    plots = [
        ('mean_score_probability_mlp', 'MLP probability'),
        ('mean_score_topsis_pi', 'PI-weighted TOPSIS'),
    ]

    for ax, (col, title) in zip(axes, plots):
        sc = ax.scatter(
            agg['Boylam'],
            agg['Enlem'],
            c=agg[col],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=10,
            marker='s',
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel('Boylam')
        ax.set_ylabel('Enlem')
        ax.set_aspect('equal', adjustable='box')

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label('Mean risk score (1-km block)')
    fig.suptitle('Full-grid wildfire priority surfaces', fontsize=15)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to: {out_path}')


if __name__ == '__main__':
    main()
