from pathlib import Path
import json
import requests
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import shape

try:
    import contextily as ctx
    HAS_CTX = True
except Exception:
    HAS_CTX = False


def download_mugla_boundary():
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": "Muğla, Türkiye",
        "format": "jsonv2",
        "polygon_geojson": 1,
        "limit": 1
    }
    headers = {
        "User-Agent": "mugla-map-script/1.0"
    }

    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()

    if not data:
        raise ValueError("Muğla boundary could not be downloaded from Nominatim.")

    geo = data[0]["geojson"]
    geom = shape(geo)
    gdf = gpd.GeoDataFrame({"name": ["Muğla"]}, geometry=[geom], crs="EPSG:4326")
    return gdf


def main():
    out = Path("figure_mugla_base_map.png")

    mugla = download_mugla_boundary()
    mugla_web = mugla.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))

    # fill + border
    mugla_web.plot(ax=ax, alpha=0.15, edgecolor="black", linewidth=2)

    if HAS_CTX:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    else:
        # no basemap case: still show boundary only
        mugla_web.boundary.plot(ax=ax, color="black", linewidth=2)

    mugla_web.boundary.plot(ax=ax, color="black", linewidth=2)

    minx, miny, maxx, maxy = mugla_web.total_bounds
    padx = (maxx - minx) * 0.05
    pady = (maxy - miny) * 0.05
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)

    ax.set_title("Muğla base map", fontsize=16)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {out.resolve()}")


if __name__ == "__main__":
    main()