from pathlib import Path
import argparse
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, shape

try:
    import contextily as ctx
    HAS_CTX = True
except Exception:
    HAS_CTX = False


def first_existing(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def download_mugla_boundary():
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": "Muğla, Türkiye",
        "format": "jsonv2",
        "polygon_geojson": 1,
        "limit": 1
    }
    headers = {"User-Agent": "mugla-risk-blocks-map/1.0"}

    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError("Muğla boundary could not be downloaded from Nominatim.")

    geom = shape(data[0]["geojson"])
    return gpd.GeoDataFrame({"name": ["Muğla"]}, geometry=[geom], crs="EPSG:4326")


def build_blocks_gdf(df, x_block_col, y_block_col, crs):
    df = df.copy()
    df[x_block_col] = pd.to_numeric(df[x_block_col], errors="coerce")
    df[y_block_col] = pd.to_numeric(df[y_block_col], errors="coerce")
    df = df.dropna(subset=[x_block_col, y_block_col])

    # x_block and y_block are block indices; convert to 1-km square polygons
    geoms = []
    for _, row in df.iterrows():
        x0 = float(row[x_block_col]) * 1000.0
        y0 = float(row[y_block_col]) * 1000.0
        geoms.append(box(x0, y0, x0 + 1000.0, y0 + 1000.0))

    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=crs)
    return gdf


def main():
    parser = argparse.ArgumentParser(
        description="Plot 1-km hotspot concentration map over Muğla."
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to risk_blocks_1000m.csv")
    parser.add_argument("--output", type=str, default="figure_risk_blocks_map.png",
                        help="Output PNG path")
    parser.add_argument("--data-crs", type=str, default="EPSG:3035",
                        help="CRS of x_block/y_block based grid system (default: EPSG:3035)")
    parser.add_argument("--color-col", type=str, default="mean_score_topsis_pi",
                        help="Column used for block coloring (default: mean_score_topsis_pi)")
    parser.add_argument("--rank-by", type=str, default="mean_score_probability_mlp",
                        help="Column used to rank and label top blocks (default: mean_score_probability_mlp)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top blocks to label (default: 5)")
    parser.add_argument("--no-basemap", action="store_true",
                        help="Disable OSM basemap")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    x_block_col = first_existing(cols, ["x_block", "block_x"])
    y_block_col = first_existing(cols, ["y_block", "block_y"])

    if x_block_col is None or y_block_col is None:
        raise ValueError(
            f"Could not detect x_block/y_block columns.\nFound columns: {cols}"
        )

    if args.color_col not in cols:
        raise ValueError(
            f"Color column '{args.color_col}' not found.\nFound columns: {cols}"
        )

    if args.rank_by not in cols:
        raise ValueError(
            f"Rank-by column '{args.rank_by}' not found.\nFound columns: {cols}"
        )

    df[args.color_col] = pd.to_numeric(df[args.color_col], errors="coerce")
    df[args.rank_by] = pd.to_numeric(df[args.rank_by], errors="coerce")
    df = df.dropna(subset=[args.color_col, args.rank_by])

    blocks_gdf = build_blocks_gdf(df, x_block_col, y_block_col, args.data_crs)
    blocks_web = blocks_gdf.to_crs(epsg=3857)

    mugla = download_mugla_boundary()
    mugla_web = mugla.to_crs(epsg=3857)

    try:
        blocks_web = gpd.clip(blocks_web, mugla_web)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 10))

    if HAS_CTX and not args.no_basemap:
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass

    # plot block surface
    blocks_web.plot(
        column=args.color_col,
        ax=ax,
        legend=True,
        cmap="viridis",
        linewidth=0.0,
        alpha=0.90,
        legend_kwds={"label": args.color_col.replace("_", " ")}
    )

    # Muğla border on top
    mugla_web.boundary.plot(ax=ax, color="black", linewidth=1.4)

    # label top-N blocks according to rank-by column
    top_blocks = blocks_web.sort_values(args.rank_by, ascending=False).head(args.top_n)
    top_blocks.boundary.plot(ax=ax, color="red", linewidth=1.8)

    for i, (_, row) in enumerate(top_blocks.iterrows(), start=1):
        cx = row.geometry.centroid.x
        cy = row.geometry.centroid.y
        ax.annotate(
            str(i),
            xy=(cx, cy),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85),
            zorder=5
        )

    minx, miny, maxx, maxy = mugla_web.total_bounds
    padx = (maxx - minx) * 0.04
    pady = (maxy - miny) * 0.04
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)

    ax.set_title("Spatial concentration of prioritized wildfire risk (1-km blocks)")
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {Path(args.output).resolve()}")
    print(f"Color column used: {args.color_col}")
    print(f"Top blocks ranked by: {args.rank_by}")
    print(f"Labeled top-N blocks: {args.top_n}")


if __name__ == "__main__":
    main()