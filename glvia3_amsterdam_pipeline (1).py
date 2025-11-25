#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GLVIA3 Townscape Attributes (TA1..TA9) – Amsterdam case
• Stable AHN WCS (rasterio only)
• CBS ELU WFS with graceful fallback to OSM land-use
• 3DBAG WFS with retries + fallback to OSM building footprints (heights from tags)
"""

from __future__ import annotations
import os, io, time, math, warnings
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import requests
from lxml import etree
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import rasterio
from rasterio.io import MemoryFile
from affine import Affine
import momepy
from scipy import stats

# ----------------------------- Settings -----------------------------
ox.settings.log_console = False
ox.settings.use_cache = True

OUT_DIR = "outputs"
PLOT_DIR = "plots"
GPKG_PATH = os.path.join(OUT_DIR, "ta_layers.gpkg")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# AOIs via OSM IDs (robust)
AOI_PRIMARY_OSMID   = "R11960504"  # Amsterdam-Centrum
AOI_SECONDARY_OSMID = "R11093212"  # IJburg

# CRS
EPSG_WGS84 = 4326
EPSG_RDNEW = 28992  # Amersfoort / RD New

# AHN WCS (stable coverage + fallback)
AHN_WCS         = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
AHN_COVERAGE_ID = "dtm_05m"  # fallback to dsm_05m if 404
AHN_PIXEL_SIZE  = 2.0        # meters (server resamples with SCALESIZE)

# 3DBAG WFS
BAG3D_WFS_BASE = "https://data.3dbag.nl/api/BAG3D/wfs"
BAG3D_WFS_CAP  = f"{BAG3D_WFS_BASE}?request=GetCapabilities"
BAG3D_LAYERTITLE_SUBSTR = "LoD 1.3 2D"

# CBS ELU WFS
CBS_ELU_WFS = "https://service.pdok.nl/cbs/landuse/wfs/v1_0"

# Amsterdam Trees
AMST_TREES_GEOJSON = "https://api.data.amsterdam.nl/v1/bomen/stamgegevens?_format=geojson"

# ----------------------------- Utils -----------------------------
def save_fig(path: str, dpi: int = 180):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def write_layer(gdf: gpd.GeoDataFrame, name: str):
    if gdf is None or gdf.empty:
        return
    try:
        gdf.to_file(GPKG_PATH, layer=name, driver="GPKG")
    except Exception as e:
        print(f"NOTE: failed to write layer {name} -> {e}")

def to_crs(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS.")
    return gdf if (gdf.crs.to_epsg() or 0) == epsg else gdf.to_crs(epsg)

def get_boundary_by_osmid(osmid: str) -> gpd.GeoDataFrame:
    gdf = ox.geocode_to_gdf(osmid, by_osmid=True)
    gdf = to_crs(gdf, EPSG_RDNEW)
    gdf["name"] = gdf.get("display_name", osmid)
    return gdf[["name", "geometry"]]

def buffered_bbox_polygon(gdf: gpd.GeoDataFrame, buffer_m: float = 500.0) -> Polygon:
    return gdf.geometry.iloc[0].buffer(buffer_m).envelope

def bbox_to_tuple(poly: Polygon) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = poly.bounds
    return (minx, miny, maxx, maxy)

def _as_str_lower(val) -> str:
    """Return a safe lowercase string for mixed types from OSM (NaN, list, bool, etc.)."""
    if isinstance(val, str):
        return val.lower()
    if isinstance(val, (list, tuple, set)):
        return " ".join(str(v) for v in val).lower()
    if val is None:
        return ""
    try:
        return str(val).lower()
    except Exception:
        return ""

# Simple HTTP GET with retries/backoff (for flaky WFS/WCS)
def http_get(url: str, params=None, timeout=60, retries=3, backoff=2.0, expect_json=False):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 404:
                r.raise_for_status()  # explicit 404
            if expect_json:
                return r.json()
            return r
        except Exception as e:
            last_err = e
            sleep_s = backoff ** i
            print(f"HTTP GET retry {i+1}/{retries} for {url} ({e}); sleeping {sleep_s:.1f}s...")
            time.sleep(sleep_s)
    if last_err:
        raise last_err

# ----------------------------- 3DBAG + OSM buildings fallback -----------------------------
def parse_wfs_typename_for_lod13() -> Optional[str]:
    try:
        r = http_get(BAG3D_WFS_CAP, timeout=30)
        xml = etree.fromstring(r.content)
        ns = {"wfs": "http://www.opengis.net/wfs", "ows": "http://www.opengis.net/ows"}
        for ft in xml.findall(".//wfs:FeatureType", ns):
            title_el = ft.find("ows:Title", ns); name_el = ft.find("wfs:Name", ns)
            if title_el is not None and name_el is not None:
                if BAG3D_LAYERTITLE_SUBSTR.lower() in (title_el.text or "").lower():
                    return name_el.text or ""
    except Exception as e:
        print("WARNING: 3DBAG WFS capabilities parse failed:", e)
    return "BAG3D:lod13"

def _parse_height_from_tags(row) -> float:
    # prefer explicit height (meters)
    for key in ["height", "building:height"]:
        v = row.get(key)
        if v is None: continue
        s = _as_str_lower(v)
        try:
            # strip units like 'm'
            s = s.replace("m","").strip()
            return float(s)
        except Exception:
            pass
    # fallback: levels * 3.0 m
    for key in ["building:levels", "levels"]:
        v = row.get(key)
        if v is None: continue
        try:
            return float(v) * 3.0
        except Exception:
            pass
    return float("nan")

def fetch_osm_buildings_fallback(aoi_poly_rd: Polygon) -> gpd.GeoDataFrame:
    """If 3DBAG is unavailable, use OSM building polygons with heights derived from tags."""
    print("Using OSM buildings fallback...")
    aoi_wgs = gpd.GeoSeries([aoi_poly_rd], crs=EPSG_RDNEW).to_crs(EPSG_WGS84).iloc[0]
    tags = {"building": True}
    g = ox.features_from_polygon(aoi_wgs, tags=tags).to_crs(EPSG_RDNEW)
    g = g[g.geometry.type.isin(["Polygon","MultiPolygon"])].copy()
    if g.empty:
        return gpd.GeoDataFrame(geometry=[], crs=EPSG_RDNEW)
    # derive height
    g["height_m"] = g.apply(_parse_height_from_tags, axis=1)
    # some datasets name levels differently; ensure column exists
    if "b3_bouwlagen" not in g.columns:
        g["b3_bouwlagen"] = np.nan
    if "oorspronkelijkbouwjaar" not in g.columns:
        g["oorspronkelijkbouwjaar"] = np.nan
    # clean tiny slivers
    g["__a__"] = g.geometry.area
    g = g[g["__a__"] > 5.0].drop(columns="__a__")
    return g

def fetch_3dbag_lod13(bbox_rd, srs="EPSG:28992", max_features=10000) -> gpd.GeoDataFrame:
    typename = parse_wfs_typename_for_lod13()
    params = dict(service="WFS", version="2.0.0", request="GetFeature",
                  typeNames=typename, outputFormat="application/json",
                  srsName=srs, bbox=f"{bbox_rd[0]},{bbox_rd[1]},{bbox_rd[2]},{bbox_rd[3]},{srs}",
                  count=max_features, startIndex=0)
    features = []
    try:
        while True:
            js = http_get(BAG3D_WFS_BASE, params=params, timeout=60, retries=3, backoff=2.0, expect_json=True)
            feats = js.get("features", [])
            if not feats: break
            features.extend(feats)
            if len(feats) < params["count"]: break
            params["startIndex"] += params["count"]; time.sleep(0.5)
    except Exception as e:
        print("WARN: 3DBAG WFS failed ->", e)
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{EPSG_RDNEW}")

    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{EPSG_RDNEW}")
    gdf = gpd.GeoDataFrame.from_features(features, crs=f"EPSG:{EPSG_RDNEW}")
    for col in ["b3_h_70p","b3_h_maaiveld","b3_bouwlagen","oorspronkelijkbouwjaar"]:
        if col not in gdf.columns: gdf[col] = np.nan
    gdf["height_m"] = gdf["b3_h_70p"] - gdf["b3_h_maaiveld"]
    return gdf

# ----------------------------- AHN WCS (stable rasterio read) -----------------------------
def _ahn_request_cov(bbox_rd, cov_id: str, pixel_size: float):
    minx, miny, maxx, maxy = bbox_rd
    width  = max(1, int((maxx - minx) / pixel_size))
    height = max(1, int((maxy - miny) / pixel_size))
    subset = f"SUBSET=x({minx},{maxx})&SUBSET=y({miny},{maxy})"
    url = (f"{AHN_WCS}?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
           f"&COVERAGEID={cov_id}&FORMAT=image/tiff&{subset}"
           f"&SCALESIZE=x({width}),y({height})")
    r = http_get(url, timeout=120, retries=3, backoff=2.0)
    if r.status_code == 404:
        raise FileNotFoundError(f"Coverage {cov_id} not found.")
    with MemoryFile(r.content) as mem:
        with mem.open() as src:
            arr = src.read(1, masked=True)
            transform: Affine = src.transform
            crs = src.crs
    return arr, transform, crs

def fetch_ahn_dtm_bbox(bbox_rd, coverage_id=AHN_COVERAGE_ID, pixel_size=AHN_PIXEL_SIZE):
    try:
        return _ahn_request_cov(bbox_rd, coverage_id, pixel_size)   # DTM
    except FileNotFoundError:
        return _ahn_request_cov(bbox_rd, "dsm_05m", pixel_size)     # DSM fallback

def slope_from_dtm(arr, transform: Affine):
    px = transform.a
    py = -transform.e
    data = np.array(arr, dtype="float32")
    data[~np.isfinite(data)] = np.nan
    gy, gx = np.gradient(data, py, px)
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    return slope, transform

# ----------------------------- CBS ELU WFS (graceful) + OSM fallback -----------------------------
def fetch_cbs_elu_bbox(bbox_rd: Tuple[float,float,float,float], srs="EPSG:28992", count=10000) -> gpd.GeoDataFrame:
    params = dict(
        service="WFS", version="2.0.0", request="GetFeature",
        typeNames="ELU:LandUseObject",
        outputFormat="application/json",
        srsName=srs,
        bbox=f"{bbox_rd[0]},{bbox_rd[1]},{bbox_rd[2]},{bbox_rd[3]},{srs}",
        count=count, startIndex=0
    )
    features = []
    try:
        while True:
            js = http_get(CBS_ELU_WFS, params=params, timeout=60, retries=2, backoff=2.0, expect_json=True)
            feats = js.get("features", [])
            if not feats: break
            features.extend(feats)
            if len(feats) < params["count"]: break
            params["startIndex"] += params["count"]; time.sleep(0.4)
    except Exception as e:
        print("WARN: CBS ELU WFS failed ->", e)
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{EPSG_RDNEW}")

    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{EPSG_RDNEW}")

    gdf = gpd.GeoDataFrame.from_features(features, crs=f"EPSG:{EPSG_RDNEW}")
    for col in ["function", "landUse", "LandUseType", "class", "functie"]:
        if col in gdf.columns:
            gdf.rename(columns={col: "landuse"}, inplace=True)
            break
    if "landuse" not in gdf.columns:
        gdf["landuse"] = None
    return gdf

def fetch_osm_landuse_fallback(aoi_poly_rd: Polygon) -> gpd.GeoDataFrame:
    aoi_wgs = gpd.GeoSeries([aoi_poly_rd], crs=EPSG_RDNEW).to_crs(EPSG_WGS84).iloc[0]
    tags = {"landuse": True, "leisure": True, "natural": True, "water": True}
    g = ox.features_from_polygon(aoi_wgs, tags=tags).to_crs(EPSG_RDNEW)
    g = g[g.geometry.type.isin(["Polygon","MultiPolygon"])].copy()
    if g.empty:
        return gpd.GeoDataFrame(geometry=[], crs=EPSG_RDNEW)
    for col in ["landuse", "natural", "leisure", "water"]:
        if col not in g.columns:
            g[col] = None
    def map_lu(row):
        lu  = _as_str_lower(row.get("landuse"))
        nat = _as_str_lower(row.get("natural"))
        le  = _as_str_lower(row.get("leisure"))
        wat = _as_str_lower(row.get("water"))
        if "water" in (lu + " " + nat + " " + wat): return "water"
        if any(k in lu for k in ["residential","industrial","commercial","retail","construction"]): return "built"
        if any(k in le for k in ["park","pitch","garden"]) or any(k in nat for k in ["wood","grass","grassland","scrub","heath","forest"]): return "green"
        if any(k in lu for k in ["meadow","grass","forest"]): return "green"
        return "other"
    g["landuse"] = g.apply(map_lu, axis=1)
    g = g.explode(index_parts=False).reset_index(drop=True)
    g["__area__"] = g.geometry.area
    g = g[g["__area__"] > 5.0].drop(columns="__area__")
    return g

# ----------------------------- OSMnx helpers -----------------------------
def fetch_pedestrian_network(aoi_poly_rd: Polygon):
    aoi_wgs = gpd.GeoSeries([aoi_poly_rd], crs=EPSG_RDNEW).to_crs(EPSG_WGS84).iloc[0]
    # try walk network, fallback to 'all' if walk fails
    try:
        G = ox.graph_from_polygon(aoi_wgs, network_type="walk", simplify=True)
    except Exception:
        G = ox.graph_from_polygon(aoi_wgs, network_type="all", simplify=True)
    G = ox.project_graph(G, to_crs=EPSG_RDNEW)
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    return nodes, edges, G

def fetch_osm_water(aoi_poly_rd: Polygon) -> gpd.GeoDataFrame:
    aoi_wgs = gpd.GeoSeries([aoi_poly_rd], crs=EPSG_RDNEW).to_crs(EPSG_WGS84).iloc[0]
    tags = {"natural": ["water"], "waterway": True}
    gdf = ox.features_from_polygon(aoi_wgs, tags).to_crs(EPSG_RDNEW)
    water = gdf[gdf.geometry.type.isin(["Polygon","MultiPolygon"])].copy()
    if not water.empty:
        water = water.dissolve().explode(index_parts=False).reset_index(drop=True)
    return water

# ----------------------------- Metrics & plotting -----------------------------
def street_enclosure_index(edges: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, radius=20.0) -> gpd.GeoDataFrame:
    # Fill missing heights using citywide median if available
    b = buildings.copy()
    if "height_m" not in b.columns:
        b["height_m"] = np.nan
    med_height = float(b["height_m"].replace([np.inf,-np.inf], np.nan).dropna().median()) if b["height_m"].notna().any() else 9.0
    b["height_m"] = b["height_m"].fillna(med_height)

    seg_buffers = edges.geometry.buffer(radius)
    idx = gpd.sjoin(
        gpd.GeoDataFrame(geometry=seg_buffers, crs=edges.crs),
        b[["height_m","geometry"]],
        predicate="intersects", how="left",
    )
    med_heights = idx.groupby(idx.index)["height_m"].median().fillna(med_height)
    ei = 2.0 * (med_heights / (2.0 * radius))
    out = edges.copy()
    out["TA1_enclosure_EI"] = ei.values
    return out

def spacematrix_indices(buildings: gpd.GeoDataFrame, aoi_poly: Polygon) -> Dict[str, float]:
    aoi_area = aoi_poly.area
    b = buildings.copy()
    b["foot_area"] = b.geometry.area
    # derive floors if not present
    if "b3_bouwlagen" in b.columns and b["b3_bouwlagen"].notna().any():
        floors = b["b3_bouwlagen"].fillna((b.get("height_m", 0.0) / 3.0))
    else:
        # attempt from height_m
        floors = (b.get("height_m", pd.Series(np.zeros(len(b))))) / 3.0
    b["floor_area"] = b["foot_area"] * floors
    gsi = b["foot_area"].sum() / aoi_area
    fsi = b["floor_area"].sum() / aoi_area
    L   = (aoi_area - b["foot_area"].sum()) / aoi_area
    return {"GSI": float(gsi), "FSI": float(fsi), "L": float(L)}

def momepy_tessellation(buildings: gpd.GeoDataFrame, streets_union_geom) -> gpd.GeoDataFrame:
    b = buildings.copy(); b["uID"]=range(len(b))
    limit = streets_union_geom.buffer(100)
    return momepy.Tessellation(b, unique_id="uID", limit=limit).tessellation

def grid_sample_points(aoi_poly: Polygon, spacing=100.0) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = aoi_poly.bounds
    xs = np.arange(minx + spacing/2, maxx, spacing)
    ys = np.arange(miny + spacing/2, maxy, spacing)
    pts = [Point(x,y) for x in xs for y in ys if aoi_poly.contains(Point(x,y))]
    return gpd.GeoDataFrame(geometry=pts, crs=f"EPSG:{EPSG_RDNEW}")

# ----------------------------- Main -----------------------------
def main():
    warnings.filterwarnings("ignore")
    metrics = {}

    # AOIs
    aoi_primary   = get_boundary_by_osmid(AOI_PRIMARY_OSMID)
    aoi_secondary = get_boundary_by_osmid(AOI_SECONDARY_OSMID)
    write_layer(aoi_primary, "aoi_primary")
    write_layer(aoi_secondary, "aoi_secondary")

    bbox_primary_poly   = buffered_bbox_polygon(aoi_primary, 500.0)
    bbox_secondary_poly = buffered_bbox_polygon(aoi_secondary, 500.0)
    bbox_primary   = bbox_to_tuple(bbox_primary_poly)
    bbox_secondary = bbox_to_tuple(bbox_secondary_poly)

    # Buildings (3DBAG with retries; fallback to OSM buildings)
    print("Downloading 3DBAG LoD1.3 (primary AOI)...")
    bldg_primary = fetch_3dbag_lod13(bbox_primary)
    if bldg_primary.empty:
        bldg_primary = fetch_osm_buildings_fallback(aoi_primary.geometry.iloc[0])
    bldg_primary = gpd.overlay(bldg_primary, aoi_primary, how="intersection")
    write_layer(bldg_primary, "buildings_primary")

    print("Downloading 3DBAG LoD1.3 (secondary AOI)...")
    bldg_secondary = fetch_3dbag_lod13(bbox_secondary)
    if bldg_secondary.empty:
        bldg_secondary = fetch_osm_buildings_fallback(aoi_secondary.geometry.iloc[0])
    bldg_secondary = gpd.overlay(bldg_secondary, aoi_secondary, how="intersection")
    write_layer(bldg_secondary, "buildings_secondary")

    # Networks
    print("Downloading OSMnx pedestrian network...")
    nodes_p, edges_p, G_p = fetch_pedestrian_network(aoi_primary.geometry.iloc[0])
    nodes_s, edges_s, G_s = fetch_pedestrian_network(aoi_secondary.geometry.iloc[0])
    write_layer(nodes_p, "net_nodes_primary"); write_layer(edges_p, "net_edges_primary")
    write_layer(nodes_s, "net_nodes_secondary"); write_layer(edges_s, "net_edges_secondary")

    # Terrain / slope (robust)
    print("Fetching AHN DTM via WCS for primary AOI...")
    try:
        dtm_arr, dtm_transform, _ = fetch_ahn_dtm_bbox(bbox_primary)
        slope_arr, slope_transform = slope_from_dtm(dtm_arr, dtm_transform)
        slope_ok = True
    except Exception as e:
        print("WARN: AHN WCS failed; TA2 will be NaN. Error:", e)
        slope_arr, slope_transform, slope_ok = None, None, False

    # Land use with fallback
    print("Fetching CBS Existing Land Use for primary AOI...")
    elu_p = fetch_cbs_elu_bbox(bbox_primary)
    if elu_p.empty:
        print("CBS ELU unavailable; using OSM land-use fallback...")
        elu_p = fetch_osm_landuse_fallback(aoi_primary.geometry.iloc[0])
    if not elu_p.empty:
        try:
            elu_p = gpd.overlay(elu_p, aoi_primary, how="intersection")
        except Exception:
            pass
    write_layer(elu_p, "landuse_primary")

    # Water
    if not elu_p.empty and "landuse" in elu_p.columns:
        water_p = elu_p[elu_p["landuse"].str.contains("water", case=False, na=False)].copy()
    else:
        water_p = gpd.GeoDataFrame(geometry=[], crs=EPSG_RDNEW)
    if water_p is None or water_p.empty:
        print("Falling back to OSM water features...")
        water_p = fetch_osm_water(aoi_primary.geometry.iloc[0])
    write_layer(water_p, "water_primary")

    # ---------- TA1 Context/Setting ----------
    print("Computing TA1 (Context/Setting)...")
    edges_ei = street_enclosure_index(edges_p, bldg_primary, radius=20.0)
    if not water_p.empty:
        w_union = unary_union(water_p.geometry.values)
        edges_ei["dist_water_m"] = edges_ei.geometry.apply(lambda g: g.distance(w_union))
    else:
        edges_ei["dist_water_m"] = np.nan
    seg_buffers_50 = edges_p.geometry.buffer(50)
    idx = gpd.sjoin(gpd.GeoDataFrame(geometry=seg_buffers_50, crs=edges_p.crs),
                    bldg_primary[["height_m","geometry"]],
                    predicate="intersects", how="left")
    std_heights = idx.groupby(idx.index)["height_m"].std()
    edges_ei["TA1_skyline_roughness"] = std_heights.values
    near = edges_ei["TA1_enclosure_EI"][edges_ei["dist_water_m"] <= 50].replace([np.inf,-np.inf],np.nan).dropna()
    far  = edges_ei["TA1_enclosure_EI"][edges_ei["dist_water_m"] > 150].replace([np.inf,-np.inf],np.nan).dropna()
    if len(near)>20 and len(far)>20:
        t_stat, p_val = stats.ttest_ind(near, far, equal_var=False, nan_policy="omit")
        eff = (near.mean()-far.mean())/(near.std(ddof=1)+1e-9)
        metrics["TA1_t_enclosure_near_vs_far"]=float(t_stat)
        metrics["TA1_p_enclosure_near_vs_far"]=float(p_val)
        metrics["TA1_d_enclosure_effect"]   =float(eff)
    plt.figure(figsize=(8,5)); edges_ei.plot(column="TA1_enclosure_EI", legend=True)
    save_fig(os.path.join(PLOT_DIR, "TA1_enclosure_map.png"))
    write_layer(edges_ei, "ta1_edges_enclosure")

    # ---------- TA2 Topography ↔ Form ----------
    print("Computing TA2 (Topography↔Form)...")
    if slope_ok:
        bcent = bldg_primary.copy(); bcent["geometry"]=bcent.centroid
        inv: Affine = ~slope_transform
        rc = [inv*(pt.x, pt.y) for pt in bcent.geometry]
        H, W = slope_arr.shape
        def sample_rc(r,c):
            r, c = int(round(r)), int(round(c))
            return slope_arr[r,c] if 0<=r<H and 0<=c<W else np.nan
        bldg_primary["slope_deg_at_centroid"] = [sample_rc(r,c) for (c,r) in rc]  # note affine order
        r1 = pd.Series(bldg_primary["slope_deg_at_centroid"]).corr(pd.Series(bldg_primary["height_m"]), method="pearson")
        if not np.isnan(r1): metrics["TA2_r_slope_vs_height"]=float(r1)
        plt.figure(figsize=(6,5))
        plt.scatter(bldg_primary["slope_deg_at_centroid"], bldg_primary["height_m"], s=6, alpha=0.5)
        plt.xlabel("Slope (deg)"); plt.ylabel("Building height (m)")
        save_fig(os.path.join(PLOT_DIR, "TA2_slope_vs_height.png"))
        write_layer(bldg_primary, "ta2_buildings_slope")
    else:
        metrics["TA2_r_slope_vs_height"] = np.nan

    # ---------- TA3 Grain & Historic patterns ----------
    print("Computing TA3 (Grain & Historic Patterns)...")
    streets_union_p = edges_p.geometry.unary_union
    tess_primary = momepy_tessellation(bldg_primary, streets_union_p)
    tess_primary["cell_area"]=tess_primary.geometry.area
    tess_primary["frontage_len"]=tess_primary.geometry.length
    write_layer(tess_primary, "ta3_tessellation_primary")
    streets_union_s = edges_s.geometry.unary_union
    tess_secondary = momepy_tessellation(bldg_secondary, streets_union_s)
    tess_secondary["cell_area"]=tess_secondary.geometry.area
    tess_secondary["frontage_len"]=tess_secondary.geometry.length
    write_layer(tess_secondary, "ta3_tessellation_secondary")
    u_stat, p_mw = stats.mannwhitneyu(tess_primary["cell_area"], tess_secondary["cell_area"], alternative="two-sided")
    metrics["TA3_u_cell_area_centrum_vs_ijburg"]=float(u_stat)
    metrics["TA3_p_cell_area_centrum_vs_ijburg"]=float(p_mw)
    plt.figure(figsize=(7,4))
    plt.hist(tess_primary["cell_area"], bins=40, alpha=0.6, label="Centrum")
    plt.hist(tess_secondary["cell_area"], bins=40, alpha=0.6, label="IJburg")
    plt.legend(); plt.xlabel("Tessellation cell area (m²)"); plt.ylabel("Count")
    save_fig(os.path.join(PLOT_DIR, "TA3_plot_area_hist.png"))

    # ---------- TA4 Layout/Scale/Density/Types ----------
    print("Computing TA4 (Layout/Scale/Density/Types)...")
    sm_primary = spacematrix_indices(bldg_primary, aoi_primary.geometry.iloc[0])
    for k,v in sm_primary.items(): metrics[f"TA4_{k}"]=v
    if "b3_bouwlagen" in bldg_primary.columns:
        r_f = pd.Series(bldg_primary["b3_bouwlagen"]).corr(pd.Series(bldg_primary["height_m"]), method="pearson")
        if not np.isnan(r_f): metrics["TA4_r_floors_vs_height"]=float(r_f)
    if "oorspronkelijkbouwjaar" in bldg_primary.columns:
        bldg_primary["period"]=pd.cut(
            bldg_primary["oorspronkelijkbouwjaar"],
            bins=[0,1900,1945,1970,1990,2010,2100],
            labels=["<1900","1900–45","1946–70","1971–90","1991–2010","2011+"]
        )
        bldg_primary["period"].value_counts().sort_index().to_csv(os.path.join(OUT_DIR, "TA4_period_counts.csv"))
    plt.figure(figsize=(5,3)); plt.title("Spacematrix summary (Centrum)")
    plt.text(0.1,0.7,f"GSI = {sm_primary['GSI']:.2f}")
    plt.text(0.1,0.5,f"FSI = {sm_primary['FSI']:.2f}")
    plt.text(0.1,0.3,f"L   = {sm_primary['L']:.2f}")
    plt.axis("off"); save_fig(os.path.join(PLOT_DIR, "TA4_spacematrix_text.png"))

    # ---------- TA5 Land-use patterns ----------
    print("Computing TA5 (Land Use)...")
    if not elu_p.empty:
        grid = grid_sample_points(aoi_primary.geometry.iloc[0], spacing=60.0)
        green_elu = elu_p[elu_p["landuse"].str.contains("groen|recrea|park|bos|forest|grass", case=False, na=False)].copy()

        if not water_p.empty:
            w_union = unary_union(water_p.geometry.values); grid["dist_water"]=grid.distance(w_union)
        else:
            grid["dist_water"]=np.inf

        buf100 = grid.buffer(100)
        idx = gpd.sjoin(gpd.GeoDataFrame(geometry=buf100, crs=grid.crs),
                        bldg_primary[["geometry"]], predicate="intersects", how="left")
        grid["bcount100"] = idx.groupby(idx.index).size()

        if not green_elu.empty:
            g_union = unary_union(green_elu.geometry.values); grid["dist_green"]=grid.distance(g_union)
        else:
            grid["dist_green"]=np.inf

        def proxy_class(row):
            if row["dist_water"]<=30: return "water"
            if row["bcount100"]>=5:   return "built"
            if row["dist_green"]<=30: return "green"
            return "other"

        grid["osm_proxy"]=grid.apply(proxy_class, axis=1)

        join = gpd.sjoin(grid, elu_p[["landuse","geometry"]], predicate="within", how="left")
        join["elu_class"]=join["landuse"].fillna("unknown")

        def elu_simple(s):
            s=str(s).lower()
            if "water" in s: return "water"
            if any(k in s for k in ["groen","park","bos","forest","grass","recrea"]): return "green"
            if any(k in s for k in ["verkeer","bebouwing","bouw","industry","woon","built","urban"]): return "built"
            return "other"

        join["elu_simple"]=join["elu_class"].apply(elu_simple)

        cats=["built","green","water","other"]
        y_true=pd.Categorical(join["elu_simple"], categories=cats)
        y_pred=pd.Categorical(join["osm_proxy"], categories=cats)
        cm=pd.crosstab(y_true, y_pred, dropna=False)
        cm.to_csv(os.path.join(OUT_DIR,"TA5_confusion_matrix.csv"))
        acc = np.diag(cm).sum()/cm.values.sum() if cm.values.sum()>0 else np.nan
        row_marg=cm.sum(axis=1).values; col_marg=cm.sum(axis=0).values
        pe=(row_marg@col_marg)/(cm.values.sum()**2) if cm.values.sum()>0 else np.nan
        kappa=(acc-pe)/(1-pe) if (1-pe)!=0 else np.nan
        metrics["TA5_accuracy_proxy_vs_elu"]=float(acc) if not np.isnan(acc) else np.nan
        metrics["TA5_kappa_proxy_vs_elu"]=float(kappa) if not np.isnan(kappa) else np.nan

        elu_p["area"]=elu_p.geometry.area
        top=(elu_p.groupby("landuse")["area"].sum().sort_values(ascending=False).head(10))
        plt.figure(figsize=(8,4)); top.plot(kind="bar"); plt.ylabel("Area (m²)")
        save_fig(os.path.join(PLOT_DIR,"TA5_elu_top10_area.png"))

    # ---------- TA6 Water contribution ----------
    print("Computing TA6 (Water contribution)...")
    if not water_p.empty:
        w_union = unary_union(water_p.geometry.values)
        edges_p["near_water_50m"] = edges_p.geometry.apply(lambda g: g.distance(w_union) <= 50.0)
        metrics["TA6_street_share_within_50m_water"] = float(edges_p["near_water_50m"].mean())
        merged = edges_ei.dropna(subset=["TA1_enclosure_EI"]).copy()
        r_w = pd.Series(merged["dist_water_m"]).corr(pd.Series(merged["TA1_enclosure_EI"]), method="pearson")
        if not np.isnan(r_w): metrics["TA6_r_distwater_vs_enclosure"]=float(r_w)
        plt.figure(figsize=(8,5))
        edges_p.plot(color="lightgray", linewidth=0.6)
        edges_p[edges_p["near_water_50m"]].plot(linewidth=1.5)
        save_fig(os.path.join(PLOT_DIR,"TA6_waterfront_segments.png"))
        write_layer(edges_p,"ta6_edges_waterfrontflag")

    # ---------- TA7 Vegetation ----------
    print("Computing TA7 (Vegetation)...")

    trees_p = gpd.GeoDataFrame()
    try:
        r = http_get(AMST_TREES_GEOJSON, timeout=15, retries=1, backoff=1.5)
        trees_p = gpd.read_file(io.BytesIO(r.content)).to_crs(EPSG_RDNEW)
        # clip to AOI bbox, then to polygon
        minx, miny, maxx, maxy = bbox_primary
        trees_p = trees_p.cx[minx:maxx, miny:maxy]
        trees_p = gpd.overlay(trees_p, aoi_primary, how="intersection")
    except Exception as e:
        print("WARN: Trees API failed ->", e)

    write_layer(trees_p, "trees_primary")

    # default zeros
    edges_p["tree_count"] = 0

    if not trees_p.empty:
        # nearest edge within 20 m
        j = gpd.sjoin_nearest(
            trees_p[["geometry"]],
            edges_p[["geometry"]],
            how="left",
            max_distance=20,
            distance_col="dist"
        )
        # robustly find the right-hand index column
        if "index_right" in j.columns:
            grp_col = "index_right"
        else:
            grp_col = next((c for c in j.columns if c.startswith("index_right")), None)

        if grp_col is not None:
            counts = j.groupby(grp_col).size()
        else:
            # fallback: within 20 m buffered edges
            buf = edges_p.copy()
            buf["geometry"] = buf.geometry.buffer(20)
            j2 = gpd.sjoin(trees_p[["geometry"]], buf, predicate="within", how="left")
            if "index_right" in j2.columns:
                counts = j2.groupby("index_right").size()
            else:
                rc = next((c for c in j2.columns if c.startswith("index_right")), None)
                counts = j2.groupby(rc).size() if rc is not None else pd.Series(dtype=int)

        edges_p["tree_count"] = edges_p.index.map(counts).fillna(0).astype(int)

    edges_p["length_m"] = edges_p.geometry.length
    edges_p["trees_per_km"] = edges_p["tree_count"] / (edges_p["length_m"] / 1000.0)
    metrics["TA7_mean_trees_per_km"] = float(
        edges_p["trees_per_km"].replace([np.inf, -np.inf], np.nan).dropna().mean()
    )
    write_layer(edges_p, "ta7_edges_trees")

    # optional correlation to green land-use
    if not elu_p.empty and not trees_p.empty:
        green = elu_p[elu_p["landuse"].str.contains(
            "groen|park|bos|forest|grass|recrea", case=False, na=False)].copy()
        if not green.empty:
            g_union = unary_union(green.geometry.values)
            edges_p["dist_green"] = edges_p.geometry.apply(lambda g: g.distance(g_union))
            r_tg = pd.Series(edges_p["trees_per_km"]).corr(
                pd.Series(edges_p["dist_green"]).map(lambda d: -d),
                method="pearson"
            )
            if not np.isnan(r_tg):
                metrics["TA7_r_trees_per_km_vs_near_green"] = float(r_tg)

    plt.figure(figsize=(8, 5))
    edges_p.plot(column="trees_per_km", legend=True)
    save_fig(os.path.join(PLOT_DIR, "TA7_trees_per_km.png"))

    # ---------- TA8 Open space & public realm ----------
    print("Computing TA8 (Open space & public realm)...")
    grid100 = grid_sample_points(aoi_primary.geometry.iloc[0], spacing=100.0)
    cells = grid100.buffer(50)
    gdf_cells = gpd.GeoDataFrame(geometry=cells, crs=grid100.crs)
    inter_area = gdf_cells.geometry.apply(lambda g: bldg_primary.intersection(g).area.sum())
    gdf_cells["open_space_index"] = 1.0 - (inter_area / (100*100))
    cc = nx.closeness_centrality(G_p.to_undirected(), distance="length")
    nodes_df = nodes_p.copy()
    nodes_df["cc"] = nodes_df.index.map(cc)
    nearest_nodes = ox.distance.nearest_nodes(
        G_p,
        X=[pt.x for pt in grid100.geometry],
        Y=[pt.y for pt in grid100.geometry]
    )
    gdf_cells["local_closeness"] = [
        nodes_df.loc[n, "cc"] if n in nodes_df.index else np.nan for n in nearest_nodes
    ]
    r_os_cc = pd.Series(gdf_cells["open_space_index"]).corr(
        pd.Series(gdf_cells["local_closeness"]), method="pearson"
    )
    if not np.isnan(r_os_cc):
        metrics["TA8_r_open_space_vs_closeness"] = float(r_os_cc)

    plt.figure(figsize=(6, 5))
    gdf_cells.plot(column="open_space_index", legend=True)
    save_fig(os.path.join(PLOT_DIR, "TA8_open_space_index.png"))
    write_layer(gdf_cells, "ta8_open_space_grid")


        # ---------- TA9 Access & connectivity ----------
    print("Computing TA9 (Access & connectivity)...")

    # helper: average edge circuity (edge length / straight-line node distance)
    def avg_edge_circuity(G):
        Gu = G.to_undirected()
        tot = 0.0
        cnt = 0
        for u, v, k, data in Gu.edges(keys=True, data=True):
            L = float(data.get("length", 0.0))
            try:
                x1, y1 = Gu.nodes[u]["x"], Gu.nodes[u]["y"]
                x2, y2 = Gu.nodes[v]["x"], Gu.nodes[v]["y"]
            except KeyError:
                continue
            d = math.hypot(x2 - x1, y2 - y1)
            if L > 0 and d > 0:
                tot += L / d
                cnt += 1
        return (tot / cnt) if cnt > 0 else np.nan

    # Intersection density (nodes with degree >= 3 per km²)
    Gu_p = G_p.to_undirected()
    deg_p = dict(Gu_p.degree())
    inter_p = sum(1 for d in deg_p.values() if d >= 3)
    area_km2_p = aoi_primary.geometry.iloc[0].area / 1e6
    metrics["TA9_intersection_density_p"] = inter_p / area_km2_p if area_km2_p > 0 else np.nan

    Gu_s = G_s.to_undirected()
    deg_s = dict(Gu_s.degree())
    inter_s = sum(1 for d in deg_s.values() if d >= 3)
    area_km2_s = aoi_secondary.geometry.iloc[0].area / 1e6
    metrics["TA9_intersection_density_s"] = inter_s / area_km2_s if area_km2_s > 0 else np.nan

    # Circuity (average across edges)
    metrics["TA9_circuity_avg_p"] = avg_edge_circuity(G_p)
    metrics["TA9_circuity_avg_s"] = avg_edge_circuity(G_s)

    # Betweenness centrality (fallback to sampling if full is heavy)
    try:
        bw = nx.betweenness_centrality(Gu_p, weight="length", normalized=True, k=None)
    except Exception:
        k = min(500, Gu_p.number_of_nodes())
        bw = nx.betweenness_centrality(Gu_p, weight="length", normalized=True, k=k)

    nodes_p["betweenness"] = nodes_p.index.map(bw)
    write_layer(nodes_p, "ta9_nodes_centrality")
    plt.figure(figsize=(8, 5))
    nodes_p.plot(column="betweenness", markersize=4, legend=True)
    save_fig(os.path.join(PLOT_DIR, "TA9_betweenness_nodes.png"))

    # Save metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(OUT_DIR,"metrics_summary.csv"), index=False)
    print("\nDone. Outputs:")
    print(f" - {OUT_DIR}/metrics_summary.csv")
    print(f" - {PLOT_DIR}/*.png")
    print(f" - {GPKG_PATH}")

if __name__ == "__main__":
    main()
