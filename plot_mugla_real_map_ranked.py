from pathlib import Path
import argparse
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import shape

try:
    import contextily as ctx
    HAS_CTX = True
except Exception:
    HAS_CTX = False


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def first_existing(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def detect_grid_columns(df):
    cols = list(df.columns)

    x_col = first_existing(cols, ["Boylam", "longitude", "lon", "lng", "x"])
    y_col = first_existing(cols, ["Enlem", "latitude", "lat", "y"])
    score_col = first_existing(
        cols,
        ["score_topsis_pi", "topsis_score", "topsis_pi", "score_topsis", "risk_score"]
    )

    if x_col is None or y_col is None or score_col is None:
        raise ValueError(
            f"Required columns could not be detected in grid CSV.\n"
            f"Found columns: {cols}"
        )

    return x_col, y_col, score_col


def detect_center_columns(df):
    cols = list(df.columns)

    x_col = first_existing(
        cols,
        [
            "snapped_Boylam", "center_Boylam", "centroid_Boylam",
            "Boylam", "longitude", "lon", "lng", "x"
        ]
    )
    y_col = first_existing(
        cols,
        [
            "snapped_Enlem", "center_Enlem", "centroid_Enlem",
            "Enlem", "latitude", "lat", "y"
        ]
    )
    rank_col = first_existing(cols, ["priority_rank", "rank", "priority"])

    if x_col is None or y_col is None:
        raise ValueError(
            f"Required columns could not be detected in centers CSV.\n"
            f"Found columns: {cols}"
        )

    return x_col, y_col, rank_col


def build_gdf(df, x_col, y_col, crs):
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs=crs
    )
    return gdf


def download_mugla_boundary():
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": "Muğla, Türkiye",
        "format": "jsonv2",
        "polygon_geojson": 1,
        "limit": 1
    }
    headers = {
        "User-Agent": "mugla-priority-map/1.0"
    }

    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()

    if not data:
        raise ValueError("Muğla boundary could not be downloaded from Nominatim.")

    geom = shape(data[0]["geojson"])
    gdf = gpd.GeoDataFrame({"name": ["Muğla"]}, geometry=[geom], crs="EPSG:4326")
    return gdf


def aggregate_scores(gdf, score_col, cell_size=1000):
    x = gdf.geometry.x
    y = gdf.geometry.y

    x_bin = np.floor(x / cell_size) * cell_size
    y_bin = np.floor(y / cell_size) * cell_size

    agg = (
        pd.DataFrame({
            "x_bin": x_bin,
            "y_bin": y_bin,
            "score": pd.to_numeric(gdf[score_col], errors="coerce")
        })
        .dropna()
        .groupby(["x_bin", "y_bin"], as_index=False)["score"]
        .mean()
    )
    return agg


def clip_if_possible(gdf, boundary_gdf):
    try:
        out = gpd.clip(gdf, boundary_gdf)
        if len(out) > 0:
            return out
    except Exception:
        pass
    return gdf


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot PI-weighted TOPSIS map over real Muğla boundary with optional ranked center overlay."
    )
    parser.add_argument("--grid-csv", type=str, required=True,
                        help="Path to all_grid_scored.csv")
    parser.add_argument("--centers-csv", type=str, default=None,
                        help="Optional path to proposed_response_centers_k*_score_topsis_pi.csv")
    parser.add_argument("--data-crs", type=str, default="EPSG:3035",
                        help="CRS of grid and center coordinates (default: EPSG:3035)")
    parser.add_argument("--output", type=str, default="figure_mugla_real_map_ranked.png",
                        help="Output PNG file")
    parser.add_argument("--cell-size", type=float, default=1000,
                        help="Aggregation cell size in meters after projection to Web Mercator")
    parser.add_argument("--no-basemap", action="store_true",
                        help="Disable OpenStreetMap basemap")
    args = parser.parse_args()

    grid_csv = Path(args.grid_csv)
    centers_csv = Path(args.centers_csv) if args.centers_csv else None
    out_path = Path(args.output)

    if not grid_csv.exists():
        raise FileNotFoundError(f"Grid CSV not found: {grid_csv}")

    if centers_csv is not None and not centers_csv.exists():
        raise FileNotFoundError(f"Centers CSV not found: {centers_csv}")

    # ---- Load grid
    grid_df = pd.read_csv(grid_csv)
    gx, gy, score_col = detect_grid_columns(grid_df)

    grid_df[score_col] = pd.to_numeric(grid_df[score_col], errors="coerce")
    grid_df = grid_df.dropna(subset=[score_col])

    grid_gdf = build_gdf(grid_df, gx, gy, args.data_crs)

    # ---- Load centers if provided
    centers_gdf = None
    rank_col = None
    if centers_csv is not None:
        centers_df = pd.read_csv(centers_csv)
        cx, cy, rank_col = detect_center_columns(centers_df)
        centers_gdf = build_gdf(centers_df, cx, cy, args.data_crs)

    # ---- Download Muğla boundary
    mugla = download_mugla_boundary()

    # ---- Reproject all to Web Mercator for plotting / basemap
    mugla_plot = mugla.to_crs(epsg=3857)
    grid_plot = grid_gdf.to_crs(epsg=3857)
    grid_plot = clip_if_possible(grid_plot, mugla_plot)

    if centers_gdf is not None:
        centers_plot = centers_gdf.to_crs(epsg=3857)
        centers_plot = clip_if_possible(centers_plot, mugla_plot)
    else:
        centers_plot = None

    # ---- Aggregate grid for a cleaner surface
    agg = aggregate_scores(grid_plot, score_col=score_col, cell_size=args.cell_size)

    # ---- Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # boundary fill + border
    mugla_plot.plot(ax=ax, alpha=0.10, edgecolor="black", linewidth=1.6)

    # basemap
    if HAS_CTX and not args.no_basemap:
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass

    # plot again on top so border stays visible
    mugla_plot.boundary.plot(ax=ax, color="black", linewidth=1.4)

    vmin = agg["score"].quantile(0.01)
    vmax = agg["score"].quantile(0.99)

    sc = ax.scatter(
        agg["x_bin"],
        agg["y_bin"],
        c=agg["score"],
        s=18,
        marker="s",
        linewidths=0,
        alpha=0.90,
        vmin=vmin,
        vmax=vmax,
        zorder=3
    )

    # ---- Proposed centers overlay with priority labels
    if centers_plot is not None and len(centers_plot) > 0:
        ax.scatter(
            centers_plot.geometry.x,
            centers_plot.geometry.y,
            marker="^",
            s=160,
            edgecolor="black",
            linewidth=0.9,
            label="Proposed response centers",
            zorder=5
        )

        if rank_col is not None:
            centers_sorted = centers_plot.sort_values(rank_col)
            for _, row in centers_sorted.iterrows():
                ax.annotate(
                    str(int(row[rank_col])),
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="black", alpha=0.85)
                )

        ax.legend(loc="upper right", frameon=True)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label("PI-weighted TOPSIS score")

    minx, miny, maxx, maxy = mugla_plot.total_bounds
    padx = (maxx - minx) * 0.04
    pady = (maxy - miny) * 0.04
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)

    ax.set_title("Muğla wildfire priority surface (PI-weighted TOPSIS)")
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {out_path.resolve()}")
    print(f"Detected score column: {score_col}")
    if centers_csv is not None:
        print(f"Centers overlay used: {centers_csv}")
        if rank_col is not None:
            print(f"Center labels taken from column: {rank_col}")
    else:
        print("Centers overlay: not used")


if __name__ == "__main__":
    main()