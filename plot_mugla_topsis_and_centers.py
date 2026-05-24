from pathlib import Path
import argparse
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    raise SystemExit(
        "This script needs geopandas and shapely.\n"
        "Install with:\n"
        "  pip install geopandas shapely pyogrio"
    )


def first_existing(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def detect_grid_columns(df: pd.DataFrame):
    cols = list(df.columns)

    x_col = first_existing(
        cols,
        ["Boylam", "longitude", "lon", "lng", "x", "X"]
    )
    y_col = first_existing(
        cols,
        ["Enlem", "latitude", "lat", "y", "Y"]
    )
    score_col = first_existing(
        cols,
        ["score_topsis_pi", "topsis_score", "topsis_pi", "score_topsis", "risk_score"]
    )

    if x_col is None or y_col is None or score_col is None:
        raise ValueError(
            f"Could not detect required columns in grid file.\n"
            f"Found columns: {cols}\n"
            f"Need coordinate columns (Boylam/Enlem or lon/lat) and a PI-TOPSIS score column."
        )

    return x_col, y_col, score_col


def detect_center_columns(df: pd.DataFrame):
    cols = list(df.columns)

    x_col = first_existing(
        cols,
        [
            "snapped_Boylam", "center_Boylam", "centroid_Boylam", "Boylam",
            "longitude", "lon", "lng", "x", "X"
        ]
    )
    y_col = first_existing(
        cols,
        [
            "snapped_Enlem", "center_Enlem", "centroid_Enlem", "Enlem",
            "latitude", "lat", "y", "Y"
        ]
    )

    if x_col is None or y_col is None:
        raise ValueError(
            f"Could not detect required columns in centers file.\n"
            f"Found columns: {cols}"
        )

    return x_col, y_col


def detect_priority_column(df: pd.DataFrame):
    cols = list(df.columns)
    pcol = first_existing(cols, ["priority_rank", "rank", "Priority", "priority"])
    return pcol


def looks_like_lonlat(x: pd.Series, y: pd.Series) -> bool:
    try:
        x_ok = x.between(-180, 180).all()
        y_ok = y.between(-90, 90).all()
        return bool(x_ok and y_ok)
    except Exception:
        return False


def read_boundary(boundary_path: Path | None):
    if boundary_path is None:
        return None
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file not found: {boundary_path}")
    gdf = gpd.read_file(boundary_path)
    if gdf.crs is None:
        # leave as-is; caller may still use it
        return gdf
    return gdf


def filter_mugla_if_possible(boundary_gdf):
    if boundary_gdf is None:
        return None

    name_candidates = [c for c in boundary_gdf.columns if c.lower() in {
        "name", "province", "il_adi", "adi", "shapeName".lower(), "adm1_name", "name_1"
    }]

    for col in name_candidates:
        vals = boundary_gdf[col].astype(str).str.lower().str.strip()
        mask = vals.isin(["muğla", "mugla"])
        if mask.any():
            return boundary_gdf.loc[mask].copy()

    return boundary_gdf


def infer_data_crs(x: pd.Series, y: pd.Series, user_crs: str | None, boundary_gdf):
    if user_crs:
        return user_crs

    if looks_like_lonlat(x, y):
        return "EPSG:4326"

    if boundary_gdf is not None and boundary_gdf.crs is not None:
        # If coordinates are not lon/lat, assume same CRS as boundary unless user overrides
        return boundary_gdf.crs

    return None


def build_geodf_from_xy(df: pd.DataFrame, x_col: str, y_col: str, crs):
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs=crs
    )
    return gdf


def clip_if_possible(gdf, boundary_gdf):
    if gdf is None or boundary_gdf is None:
        return gdf
    try:
        if gdf.crs is not None and boundary_gdf.crs is not None and gdf.crs != boundary_gdf.crs:
            gdf = gdf.to_crs(boundary_gdf.crs)
        clipped = gpd.clip(gdf, boundary_gdf)
        if len(clipped) > 0:
            return clipped
    except Exception:
        pass
    return gdf


def choose_cell_size(gdf, user_cell_size):
    if user_cell_size is not None:
        return user_cell_size

    if gdf.crs is not None and str(gdf.crs).upper().endswith("4326"):
        return 0.01  # about ~1 km-ish visual aggregation
    return 1000.0   # 1 km if projected coordinates


def aggregate_scores(gdf, score_col, cell_size):
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


def plot_map(grid_gdf, score_col, centers_gdf, boundary_gdf, output_path, title, cell_size):
    fig, ax = plt.subplots(figsize=(10, 10))

    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, linewidth=1.2, color="black")

    agg = aggregate_scores(grid_gdf, score_col, cell_size)

    vmin = agg["score"].quantile(0.01)
    vmax = agg["score"].quantile(0.99)

    sc = ax.scatter(
        agg["x_bin"],
        agg["y_bin"],
        c=agg["score"],
        s=14,
        marker="s",
        linewidths=0,
        alpha=0.95,
        vmin=vmin,
        vmax=vmax,
    )

    if centers_gdf is not None and len(centers_gdf) > 0:
        ax.scatter(
            centers_gdf.geometry.x,
            centers_gdf.geometry.y,
            marker="^",
            s=150,
            edgecolor="black",
            linewidth=0.8,
            label="Proposed response centers",
            zorder=5,
        )

        pcol = detect_priority_column(centers_gdf)
        if pcol is not None:
            centers_sorted = centers_gdf.sort_values(pcol)
            for _, row in centers_sorted.iterrows():
                ax.annotate(
                    str(int(row[pcol])),
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=9,
                    weight="bold",
                    zorder=6,
                )
        ax.legend(loc="upper right", frameon=True)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label("PI-weighted TOPSIS score")

    if boundary_gdf is not None:
        minx, miny, maxx, maxy = boundary_gdf.total_bounds
    else:
        minx, miny, maxx, maxy = grid_gdf.total_bounds

    padx = (maxx - minx) * 0.03
    pady = (maxy - miny) * 0.03
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)

    xlabel = "Longitude" if (grid_gdf.crs is not None and str(grid_gdf.crs).upper().endswith("4326")) else "X"
    ylabel = "Latitude" if (grid_gdf.crs is not None and str(grid_gdf.crs).upper().endswith("4326")) else "Y"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {output_path}")


def auto_find_file(base: Path, candidates):
    for c in candidates:
        p = base / c
        if p.exists():
            return p

        matches = list(base.glob(c))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot PI-weighted TOPSIS map and optionally overlay proposed response centers."
    )
    parser.add_argument("--grid-csv", type=str, default=None,
                        help="Path to all_grid_scored.csv")
    parser.add_argument("--centers-csv", type=str, default=None,
                        help="Optional path to proposed_response_centers_k*_score_topsis_pi.csv")
    parser.add_argument("--boundary", type=str, default=None,
                        help="Optional Muğla boundary file (.geojson or .shp)")
    parser.add_argument("--data-crs", type=str, default=None,
                        help="Optional CRS of grid/center coordinates, e.g. EPSG:4326 or EPSG:32635")
    parser.add_argument("--cell-size", type=float, default=None,
                        help="Optional aggregation cell size. Default: auto (0.01 for lon/lat, 1000 for projected)")
    parser.add_argument("--output", type=str, default="figure_mugla_topsis_and_centers.png",
                        help="Output PNG path")
    args = parser.parse_args()

    base = Path.cwd()

    grid_csv = Path(args.grid_csv) if args.grid_csv else auto_find_file(
        base,
        [
            "outputs_full_compare/all_grid_scored.csv",
            "all_grid_scored.csv"
        ]
    )
    centers_csv = Path(args.centers_csv) if args.centers_csv else auto_find_file(
        base,
        [
            "outputs_full_compare/proposed_response_centers_k*_score_topsis_pi.csv",
            "proposed_response_centers_k*_score_topsis_pi.csv"
        ]
    )
    boundary_path = Path(args.boundary) if args.boundary else auto_find_file(
        base,
        [
            "mugla_boundary.geojson",
            "mugla.geojson",
            "mugla_boundary.shp",
            "mugla.shp"
        ]
    )

    if grid_csv is None or not grid_csv.exists():
        raise FileNotFoundError(
            "Could not find all_grid_scored.csv.\n"
            "Pass it explicitly with --grid-csv or place it in the current folder / outputs_full_compare."
        )

    grid_df = pd.read_csv(grid_csv)
    gx, gy, score_col = detect_grid_columns(grid_df)
    grid_df[gx] = pd.to_numeric(grid_df[gx], errors="coerce")
    grid_df[gy] = pd.to_numeric(grid_df[gy], errors="coerce")
    grid_df[score_col] = pd.to_numeric(grid_df[score_col], errors="coerce")
    grid_df = grid_df.dropna(subset=[gx, gy, score_col])

    boundary_gdf = read_boundary(boundary_path) if boundary_path is not None else None
    boundary_gdf = filter_mugla_if_possible(boundary_gdf)

    data_crs = infer_data_crs(grid_df[gx], grid_df[gy], args.data_crs, boundary_gdf)
    grid_gdf = build_geodf_from_xy(grid_df, gx, gy, data_crs)
    grid_gdf = clip_if_possible(grid_gdf, boundary_gdf)

    centers_gdf = None
    if centers_csv is not None and centers_csv.exists():
        centers_df = pd.read_csv(centers_csv)
        cx, cy = detect_center_columns(centers_df)
        centers_df[cx] = pd.to_numeric(centers_df[cx], errors="coerce")
        centers_df[cy] = pd.to_numeric(centers_df[cy], errors="coerce")
        centers_df = centers_df.dropna(subset=[cx, cy])
        centers_gdf = build_geodf_from_xy(centers_df, cx, cy, data_crs)
        centers_gdf = clip_if_possible(centers_gdf, boundary_gdf)

    cell_size = choose_cell_size(grid_gdf, args.cell_size)

    plot_map(
        grid_gdf=grid_gdf,
        score_col=score_col,
        centers_gdf=centers_gdf,
        boundary_gdf=boundary_gdf,
        output_path=Path(args.output),
        title="Muğla wildfire priority surface (PI-weighted TOPSIS)",
        cell_size=cell_size,
    )

    print(f"Detected score column: {score_col}")
    print(f"Grid CSV: {grid_csv}")
    if centers_csv is not None and centers_csv.exists():
        print(f"Centers CSV: {centers_csv}")
    else:
        print("Centers overlay: not used")
    if boundary_path is not None and boundary_path.exists():
        print(f"Boundary file: {boundary_path}")
    else:
        print("Boundary file: not used (plotted on data extent only)")


if __name__ == "__main__":
    main()