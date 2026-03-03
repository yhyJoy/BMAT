"""
Building Line-of-Sight Visibility Analysis

For each street-view panorama point, determines whether the target building
is visible by casting sight lines to the building centroid and boundary vertices.

Input panorama file (.p pickle) must contain:
    sample_id, pano_lon, pano_lat, path (image file path)

Output CSV columns:
    sample_id, pano_lon, pano_lat, img_path, centerline_visible

Usage:
    python obstruct_analysis.py --param "China/Hong Kong" --meta_root ../data/csv --shp_root ../data/shp/
"""

import os
import argparse

import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Point, LineString, Polygon, MultiPolygon


def judge_visible(sample_id, pano_lon, pano_lat, id2center, sindex, gdf_other):
    """
    Determine whether a building is visible from a panorama point.

    Step 1: cast sight line to building centroid → visible if unobstructed.
    Step 2: if centroid is blocked, cast sight lines to all boundary vertices
            → visible if any vertex is unobstructed (partial visibility counts).
    Step 3: not visible if every sight line is blocked.
    """
    if sample_id not in id2center or pd.isna(pano_lon) or pd.isna(pano_lat):
        return False

    pano_point = Point(pano_lon, pano_lat)
    center     = id2center[sample_id]

    # Narrow candidates to buildings within the sight-line bounding box
    my_line       = LineString([pano_point, center])
    candidate_idx = list(sindex.intersection(my_line.bounds))
    candidate_gdf = gdf_other.iloc[candidate_idx]

    target_row = candidate_gdf[candidate_gdf["id_str"] == sample_id]
    if target_row.empty:
        return False
    target_poly = target_row.geometry.iloc[0]

    def is_blocked(line):
        """Return True if `line` is crossed by any building other than the target."""
        for poly_row in candidate_gdf.itertuples():
            this_id = str(getattr(poly_row, "id_str"))
            poly    = poly_row.geometry
            if poly is None or not poly.is_valid or this_id == sample_id:
                continue
            if poly.contains(pano_point):                  # camera is inside this polygon → skip
                continue
            if target_poly.within(poly) and target_poly.contains(center):  # shell polygon → skip
                continue
            if poly.within(target_poly):                   # polygon nested inside target → skip
                continue
            if line.crosses(poly):                         # genuine obstruction found
                return True
        return False

    # Step 1: centroid sight line
    if not is_blocked(LineString([pano_point, center])):
        return True

    # Step 2: boundary vertex sight lines
    boundary_points = []
    if isinstance(target_poly, Polygon):
        boundary_points = [Point(x, y) for x, y in target_poly.exterior.coords]
    elif isinstance(target_poly, MultiPolygon):
        boundary_points = [Point(x, y) for part in target_poly.geoms
                           for x, y in part.exterior.coords]
    else:
        return False  # unsupported geometry type

    for pt in boundary_points:
        if not is_blocked(LineString([pano_point, pt])):
            return True  # at least one vertex is visible

    return False


def process_city(pkl_path, shp_path, output_csv,
                 building_id_col="id", save_every=10000, verbose=True):
    """
    Run visibility analysis for all panorama records in pkl_path.
    Supports checkpoint resumption: already-processed sample_ids are skipped.
    """
    # Load building footprints and build spatial index
    gdf_building = gpd.read_file(shp_path).to_crs("EPSG:4326")
    gdf_building["id_str"] = gdf_building[building_id_col].astype(str)
    id2center = {str(row[building_id_col]): row.geometry.centroid
                 for _, row in gdf_building.iterrows() if row.geometry is not None}
    sindex = gdf_building.sindex

    # Load panorama records and drop rows with missing image paths
    ext = os.path.splitext(pkl_path)[1].lower()
    df_pano = pd.read_pickle(pkl_path) if ext in (".p", ".pkl") else pd.read_csv(pkl_path)
    mask    = (df_pano["img_path"].isna() |
               df_pano["img_path"].astype(str).str.strip().str.lower().isin(["none", "nan", ""]))
    df_pano = df_pano[~mask].copy()
    df_pano = df_pano[df_pano["img_path"].apply(os.path.exists)]  # keep only existing files
    print(f"[INFO] {len(gdf_building)} buildings | {len(df_pano)} valid panorama records")

    # Restore checkpoint if output file already exists
    processed_ids, all_results = set(), []
    if os.path.exists(output_csv):
        old_df        = pd.read_csv(output_csv)
        processed_ids = set(old_df["sample_id"].astype(str))
        all_results   = old_df.to_dict("records")
        print(f"[INFO] Resuming — {len(processed_ids)} records already done.")

    result_list = []
    iterator    = tqdm(df_pano.iterrows(), total=len(df_pano), desc="Analysing") if verbose \
                  else df_pano.iterrows()

    for _, row in iterator:
        sample_id = str(row["sample_id"])
        if sample_id in processed_ids:
            continue

        visible = judge_visible(sample_id, row["pano_lon"], row["pano_lat"],
                                id2center, sindex, gdf_building)
        result_list.append({
            "sample_id":          sample_id,
            "pano_lon":           row["pano_lon"],
            "pano_lat":           row["pano_lat"],
            "img_path":           row["img_path"],
            "centerline_visible": visible,
        })

        # Periodic save to avoid losing progress on long runs
        if (len(result_list) + len(all_results)) % save_every == 0:
            merged = all_results + result_list
            pd.DataFrame(merged) \
              .sort_values("sample_id", key=lambda x: x.astype(int)) \
              .to_csv(output_csv, index=False)
            all_results, result_list = merged, []

    # Final save
    if result_list:
        merged = all_results + result_list
        pd.DataFrame(merged) \
          .sort_values("sample_id", key=lambda x: x.astype(int)) \
          .to_csv(output_csv, index=False)

    print(f"[DONE] Saved to {output_csv}")


def run_batch(param, meta_root, shp_root,
              output_suffix="_building_obstruct.csv", save_every=10000, verbose=True):
    """Dispatch analysis by 'all' | '{region}' | '{region}/{city}'."""

    def _find_shp(region, city):
        # Accept {city}.shp / {city}_buildings.shp and .gpkg equivalents
        folder = os.path.join(shp_root, region, city)
        for stem in (city, f"{city}_buildings"):
            for ext in (".shp", ".gpkg"):
                p = os.path.join(folder, f"{stem}{ext}")
                if os.path.exists(p):
                    return p
        return None

    def _run(region, city):
        city_dir = os.path.join(meta_root, region, city)
        if not os.path.isdir(city_dir):
            return

        # Find panorama pickle file (pano_*.p)
        pkl_files = [f for f in os.listdir(city_dir)
                    if f.startswith("meta_") and f.endswith((".p", ".csv"))]
        if not pkl_files:
            print(f"[SKIP] {region}/{city}: no meta_*.p or meta_*.csv file found.")
            return

        shp_path = _find_shp(region, city)
        if not shp_path:
            print(f"[SKIP] {region}/{city}: no shapefile found.")
            return

        pkl_path   = os.path.join(city_dir, pkl_files[0])
        output_csv = os.path.join(city_dir, f"{city}{output_suffix}")
        print(f"\n[INFO] Processing {region}/{city}")
        try:
            process_city(pkl_path, shp_path, output_csv, save_every=save_every, verbose=verbose)
        except Exception as e:
            print(f"[ERROR] {region}/{city}: {e}")

    parts = param.strip().split("/")
    if param.lower() == "all":
        for region in sorted(os.listdir(meta_root)):
            if not os.path.isdir(os.path.join(meta_root, region)):
                continue
            for city in sorted(os.listdir(os.path.join(meta_root, region))):
                _run(region, city)
    elif len(parts) == 1:
        for city in sorted(os.listdir(os.path.join(meta_root, parts[0]))):
            _run(parts[0], city)
    elif len(parts) == 2:
        _run(parts[0], parts[1])
    else:
        print(f"[ERROR] Unsupported --param format: '{param}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Building line-of-sight visibility analysis.")
    parser.add_argument("--param",      type=str, required=True,
                        help="'all' | '{region}' | '{region}/{city}'")
    parser.add_argument("--meta_root",  type=str, required=True,
                        help="Root dir containing pano_*.p pickle files")
    parser.add_argument("--shp_root",   type=str, required=True,
                        help="Root dir containing building shapefiles")
    parser.add_argument("--output_suffix", type=str, default="_building_obstruct.csv")
    parser.add_argument("--save_every",    type=int, default=10000)
    parser.add_argument("--no_verbose", action="store_true")
    args = parser.parse_args()

    run_batch(args.param, args.meta_root, args.shp_root,
              args.output_suffix, args.save_every, not args.no_verbose)