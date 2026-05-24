from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

# Windows + joblib/loky uyarısını azaltmak için.
#os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

from src.location import propose_response_centers


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk sıralamasından yangın müdahale merkezi adayları üretir.")
    parser.add_argument("--ranked-csv", required=True, help="all_grid_scored.csv veya benzeri çıktı")
    parser.add_argument("--k", type=int, default=10, help="Önerilecek merkez sayısı")
    parser.add_argument("--top-n", type=int, default=5000, help="En riskli kaç hücre aday kümede kullanılacak")
    parser.add_argument("--weight-col", default="score_topsis_pi", help="Kümelendirmede ağırlık olarak hangi skor sütunu kullanılacak")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    ranked = pd.read_csv(args.ranked_csv)
    if args.weight_col not in ranked.columns:
        raise ValueError(f"Seçilen score sütunu bulunamadı: {args.weight_col}")
    out = propose_response_centers(ranked_df=ranked, k=args.k, weight_col=args.weight_col, top_n=args.top_n)

    output_csv = Path(args.output_csv) if args.output_csv else Path(args.ranked_csv).with_name(f"proposed_response_centers_k{args.k}_{args.weight_col}.csv")
    out.to_csv(output_csv, index=False)
    print(f"Merkez adayları kaydedildi: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
