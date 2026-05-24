from pathlib import Path
import argparse
import json
import math
import requests
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

try:
    import contextily as cx
    HAS_CONTEXTILY = True
except Exception:
    HAS_CONTEXTILY = False


def first_existing(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def detect_grid_cols(df):
    cols = list(df.columns)
    x = first_existing(cols, ["Boylam", "x", "X", "lon", "longitude"])
    y = first_existing(cols, ["Enlem", "y", "Y", "lat", "latitude"])
    score = first_existing(cols, ["score_topsis_pi", "topsis_score", "score_topsis", "topsis_pi"])
    if x is None or y is None or score is None:
        raise ValueError(f"Could not detect required columns in grid CSV. Found: {cols}")
    return x, y, score


def detect_center_cols(df):
    cols = list(df.columns)
    x = first_existing(cols, [
        "snapped_Boylam", "center_Boylam", "centroid_Boylam", "Boylam", "x", "X", "lon", "longitude"
    ])
    y = first_existing(cols, [
        "snapped_Enlem", "center_Enlem", "centroid_Enlem", "Enlem", "y", "Y", "lat", "latitude"
    ])
    rank = first_existing(cols, ["priority_rank", "rank", "priority"])
    if x is None or y is None:
        raise ValueError(f"Could not detect required columns in centers CSV. Found: {cols}")
    return x, y, rank


def download_mugla_boundary(cache_path: Path) -> Path:
    if cache_path.exists():
        return cache_path

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": "Muğla, Türkiye",
        "format": "jsonv2",
        "polygon_geojson": 1,
        "limit": 1,
    }
    headers = {
        "User-Agent": "mugla-wildfire-map/1.0 (research-use)"
    }
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError("Could not download Muğla boundary from Nominatim.")

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"display_name": data[0].get("display_name", "Muğla")},
                "geometry": data[0]["geojson"],
            }
        ],
    }
    cache_path.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    return cache_path


def aggregate_to_1km(grid_gdf_3035, score_col):
    d = grid_gdf_3035.copy()
    d["x_bin"] = (np.floor(d.geometry.x / 1000) * 1000).astype(int)
    d["y_bin"] = (np.floor(d.geometry.y / 1000) * 1000).astype(int)
    agg = d.groupby(["x_bin", "y_bin"], as_index=False)[score_col].mean()
    agg["geometry"] = [Point(x, y) for x, y in zip(agg["x_bin"], agg["y_bin"])]
    return gpd.GeoDataFrame(agg, geometry="geometry", crs="EPSG:3035")


def main():
    parser = argparse.ArgumentParser(description="Plot PI-weighted TOPSIS on a real Muğla map.")
    parser.add_argument("--grid-csv", default=r".\outputs_full_compare\all_grid_scored.csv")
    parser.add_argument("--centers-csv", default=r".\outputs_full_compare\proposed_response_centers_k10_score_topsis_pi.csv")
    parser.add_argument("--output", default=r".\figure_mugla_real_map.png")
    parser.add_argument("--boundary-cache", default=r".\mugla_boundary_from_nominatim.geojson")
    parser.add_argument("--data-crs", default="EPSG:3035", help="Projected CRS of Boylam/Enlem. Default: EPSG:3035")
    parser.add_argument("--no-centers", action="store_true")
    parser.add_argument("--no-basemap", action="store_true")
    args = parser.parse_args()

    grid_csv = Path(args.grid_csv)
    centers_csv = Path(args.centers_csv)
    output = Path(args.output)
    boundary_cache = Path(args.boundary_cache)

    if not grid_csv.exists():
        raise FileNotFoundError(f"Grid CSV not found: {grid_csv}")

    # 1) Load full-grid scores
    grid_df = pd.read_csv(grid_csv)
    x_col, y_col, score_col = detect_grid_cols(grid_df)
    grid_df = grid_df[[x_col, y_col, score_col]].copy()
    grid_df[x_col] = pd.to_numeric(grid_df[x_col], errors="coerce")
    grid_df[y_col] = pd.to_numeric(grid_df[y_col], errors="coerce")
    grid_df[score_col] = pd.to_numeric(grid_df[score_col], errors="coerce")
    grid_df = grid_df.dropna()

    grid_gdf = gpd.GeoDataFrame(
        grid_df,
        geometry=gpd.points_from_xy(grid_df[x_col], grid_df[y_col]),
        crs=args.data_crs,
    )

    # 2) Aggregate to 1 km for cleaner map
    grid_agg = aggregate_to_1km(grid_gdf.to_crs("EPSG:3035"), score_col)

    # 3) Download or reuse real Muğla boundary
    boundary_geojson = download_mugla_boundary(boundary_cache)
    mugla = gpd.read_file(boundary_geojson).to_crs("EPSG:3035")

    # 4) Clip aggregated grid to Muğla
    try:
        grid_agg = gpd.clip(grid_agg, mugla)
    except Exception:
        pass

    # 5) Centers overlay (optional)
    centers_gdf = None
    if (not args.no_centers) and centers_csv.exists():
        centers_df = pd.read_csv(centers_csv)
        cx_col, cy_col, rank_col = detect_center_cols(centers_df)
        centers_df[cx_col] = pd.to_numeric(centers_df[cx_col], errors="coerce")
        centers_df[cy_col] = pd.to_numeric(centers_df[cy_col], errors="coerce")
        centers_df = centers_df.dropna(subset=[cx_col, cy_col])
        centers_gdf = gpd.GeoDataFrame(
            centers_df,
            geometry=gpd.points_from_xy(centers_df[cx_col], centers_df[cy_col]),
            crs=args.data_crs,
        ).to_crs("EPSG:3035")
        try:
            centers_gdf = gpd.clip(centers_gdf, mugla)
        except Exception:
            pass
    
    # 6) Reproject everything to Web Mercator for real-map basemap
    mugla_3857 = mugla.to_crs("EPSG:3857")
    grid_3857 = grid_agg.to_crs("EPSG:3857")
    centers_3857 = centers_gdf.to_crs("EPSG:3857") if centers_gdf is not None else None

    # 7) Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    mugla_3857.boundary.plot(ax=ax, linewidth=1.2, color="black")

    vmin = grid_3857[score_col].quantile(0.01)
    vmax = grid_3857[score_col].quantile(0.99)

    sc = ax.scatter(
        grid_3857.geometry.x,
        grid_3857.geometry.y,
        c=grid_3857[score_col],
        s=18,
        marker="s",
        linewidths=0,
        alpha=0.95,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
    )

    if centers_3857 is not None and len(centers_3857) > 0:
        ax.scatter(
            centers_3857.geometry.x,
            centers_3857.geometry.y,
            marker="^",
            s=140,
            edgecolor="black",
            linewidth=0.8,
            label="Proposed response centers",
            zorder=5,
        )
        if "priority_rank" in centers_3857.columns:
            tmp = centers_3857.sort_values("priority_rank")
            for _, row in tmp.iterrows():
                ax.annotate(
                    str(int(row["priority_rank"])),
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=9,
                    weight="bold",
                    zorder=6,
                )
        ax.legend(loc="upper right")

    if HAS_CONTEXTILY and (not args.no_basemap):
        try:
            cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass

    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label("PI-weighted TOPSIS score")

    ax.set_title("Muğla wildfire priority surface (PI-weighted TOPSIS)")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {output.resolve()}")
    print(f"Used data CRS: {args.data_crs}")
    print(f"Used score column: {score_col}")
    print(f"Boundary cache: {boundary_geojson.resolve()}")


if __name__ == "__main__":
    main()
