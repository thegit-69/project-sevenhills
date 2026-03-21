"""
Roof crop extraction for EfficientNetV2-S classifier.
Reads Built_Up_Area shapefile, crops each building from ORI,
saves to class folders: 1=RCC, 2=Tiled, 3=Tin, 4=Others

Output structure:
  data/roof_crops/train/1_RCC/BADETUMNAR_001.tif
  data/roof_crops/train/2_Tiled/BADETUMNAR_002.tif
  data/roof_crops/train/3_Tin/BADETUMNAR_003.tif
  data/roof_crops/train/4_Others/BADETUMNAR_004.tif

Run: python src/preprocessing/crop_roofs.py
"""

import logging
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from shapely.geometry import mapping, box
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    # Village list — same as tiling script
    # Each entry: (village_name, ori_path, shp_path, working_crs, split)
   "villages": [
    # ── CG Villages (share one SHP file) ──
    {
        "name":        "BADETUMNAR",
        "state":       "CG",
        "split":       "train",
        "ori_path":    "/mnt/e/datasets/CG/BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO.tif",
        "shp_path":    "/mnt/e/datasets/CG/CG_shp-file/shp-file/Built_Up_Area_type.shp",
        "working_crs": "EPSG:32644",
    },
    {
        "name":        "MURDANDA",
        "state":       "CG",
        "split":       "train",
        "ori_path":    "/mnt/e/datasets/CG/MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO.tif",
        "shp_path":    "/mnt/e/datasets/CG/CG_shp-file/shp-file/Built_Up_Area_type.shp",
        "working_crs": "EPSG:32644",
    },
    {
        "name":        "NAGUL",
        "state":       "CG",
        "split":       "train",
        "ori_path":    "/mnt/e/datasets/CG/NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO.tif",
        "shp_path":    "/mnt/e/datasets/CG/CG_shp-file/shp-file/Built_Up_Area_type.shp",
        "working_crs": "EPSG:32644",
    },
    {
        "name":        "SAMLUR",
        "state":       "CG",
        "split":       "train",
        "ori_path":    "/mnt/e/datasets/CG/SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO.tif",
        "shp_path":    "/mnt/e/datasets/CG/CG_shp-file/shp-file/Built_Up_Area_type.shp",
        "working_crs": "EPSG:32644",
    },
    {
        "name":        "KUTRU",
        "state":       "CG",
        "split":       "val",
        "ori_path":    "/mnt/e/datasets/CG/KUTRU_Converted.tif",
        "shp_path":    "/mnt/e/datasets/CG/CG_shp-file/shp-file/Built_Up_Area_type.shp",
        "working_crs": "EPSG:32644",
    },

    # ── PB Villages (share one SHP file) ──
    {
        "name":        "NADALA",
        "state":       "PB",
        "split":       "train",
        "ori_path":    "/mnt/e/datasets/PB_training_dataSet_shp_file/28996_NADALA_ORTHO.tif",
        "shp_path":    "/mnt/e/datasets/PB_training_dataSet_shp_file/shp-file/Built_Up_Area_typ.shp",
        "working_crs": "EPSG:32643",
    },
    {
        "name":        "PINDORI",
        "state":       "PB",
        "split":       "train",
        "ori_path":    "/mnt/e/datasets/PB_training_dataSet_shp_file/PINDORI MAYA SINGH-TUGALWAL_28456_ortho.tif",
        "shp_path":    "/mnt/e/datasets/PB_training_dataSet_shp_file/shp-file/Built_Up_Area_typ.shp",
        "working_crs": "EPSG:32643",
    },
    {
        "name":        "TIMMOWAL",
        "state":       "PB",
        "split":       "train",
        "ori_path":    "/mnt/e/datasets/PB_training_dataSet_shp_file/TIMMOWAL_37695_ORI.tif",
        "shp_path":    "/mnt/e/datasets/PB_training_dataSet_shp_file/shp-file/Built_Up_Area_typ.shp",
        "working_crs": "EPSG:32643",
    },
    {
        "name":        "FATTU",
        "state":       "PB",
        "split":       "train",
        "ori_path":    "/mnt/e/datasets/PB_training_dataSet_shp_file/37458_fattu_bhila_ortho_3857.tif",
        "shp_path":    "/mnt/e/datasets/PB_training_dataSet_shp_file/shp-file/Built_Up_Area_typ.shp",
        "working_crs": "EPSG:32643",
    },
    {
        "name":        "BAGGA",
        "state":       "PB",
        "split":       "val",
        "ori_path":    "/mnt/e/datasets/PB_training_dataSet_shp_file/37774_bagga ortho_3857.tif",
        "shp_path":    "/mnt/e/datasets/PB_training_dataSet_shp_file/shp-file/Built_Up_Area_typ.shp",
        "working_crs": "EPSG:32643",
    },
    ],
    # Output root directory
    "output_dir": "/mnt/e/datasets/roof_crops",

    # Padding around each building in pixels
    # Gives the model context around the roof edges
    "padding_px": 10,

    # Minimum building area in square metres
    # Skip tiny fragments that are annotation errors
    "min_area_sqm": 8.0,

    # Remarks to exclude
    "exclude_remarks": ["under construction", "UNDER CONSTRUCTION"],

    # Roof type mapping
    "class_dirs": {
        1: "1_RCC",
        2: "2_Tiled",
        3: "3_Tin",
        4: "4_Others",
    },
}

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def setup_output_dirs(output_dir: str, split: str) -> dict:
    """Creates class folders and returns path dict."""
    paths = {}
    for val, dirname in CONFIG["class_dirs"].items():
        p = Path(output_dir) / split / dirname
        p.mkdir(parents=True, exist_ok=True)
        paths[val] = p
    return paths


def load_buildings(shp_path: str, working_crs: str,
                   village_name: str) -> gpd.GeoDataFrame:
    """
    Loads building shapefile, filters to this village,
    reprojects, validates, and removes under-construction.
    """
    gdf = gpd.read_file(shp_path)

    # CG SHP covers multiple villages — use all of them
    # Village filter not needed since all belong to this ORI
    log.info(f"Villages in SHP: {gdf['Village_Na'].value_counts().to_dict()}")
    # Keep all buildings — the ORI covers all villages in this SHP

    # Reproject to working CRS
    gdf = gdf.to_crs(working_crs)

    # Explode MultiPolygons
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    # Remove under construction
    if "Remarks" in gdf.columns:
        before = len(gdf)
        uc_mask = gdf["Remarks"].str.upper().str.contains(
            "UNDER CONSTRUCTION", na=False
        )
        gdf = gdf[~uc_mask].reset_index(drop=True)
        removed = before - len(gdf)
        if removed > 0:
            log.info(f"Removed {removed} under-construction buildings")

    # Remove buildings without valid Roof_type
    if "Roof_type" in gdf.columns:
        valid_types = [1, 2, 3, 4]
        gdf = gdf[gdf["Roof_type"].isin(valid_types)].reset_index(drop=True)
    else:
        log.error("Roof_type column not found in shapefile!")
        return gpd.GeoDataFrame()

    # Remove tiny buildings (annotation errors)
    if "Area_Sqm" in gdf.columns:
        before = len(gdf)
        gdf = gdf[
            gdf["Area_Sqm"] >= CONFIG["min_area_sqm"]
        ].reset_index(drop=True)
        log.info(
            f"Removed {before - len(gdf)} buildings "
            f"smaller than {CONFIG['min_area_sqm']} m²"
        )

    # Remove invalid geometries
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]

    log.info(f"Buildings loaded: {len(gdf)}")

    # Log class distribution
    counts = gdf["Roof_type"].value_counts().sort_index()
    class_names = {1: "RCC", 2: "Tiled", 3: "Tin", 4: "Others"}
    for rtype, count in counts.items():
        log.info(
            f"  Roof type {rtype} ({class_names.get(rtype,'?')}): "
            f"{count} buildings ({count/len(gdf)*100:.1f}%)"
        )

    return gdf


def crop_building(src: rasterio.DatasetReader,
                  geom,
                  padding_px: int) -> np.ndarray | None:
    """
    Crops a single building from the open rasterio dataset.
    Adds padding around the bounding box.
    Returns RGB array (3, H, W) or None if crop fails.
    """
    try:
        # Get pixel size from transform
        px_size = abs(src.transform.a)
        padding_m = padding_px * px_size

        # Expand bounding box by padding
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        padded_bounds = (
            bounds[0] - padding_m,
            bounds[1] - padding_m,
            bounds[2] + padding_m,
            bounds[3] + padding_m,
        )

        padded_geom = box(*padded_bounds)

        # Crop from raster
        out_image, out_transform = rio_mask(
            src,
            [mapping(padded_geom)],
            crop=True,
            nodata=0,
            filled=True,
        )

        # Skip if crop is empty or too small
        if out_image is None:
            return None
        if out_image.shape[1] < 4 or out_image.shape[2] < 4:
            return None

        # Skip mostly-black crops (outside flight boundary)
        if np.mean(out_image) < 5:
            return None

        return out_image

    except Exception as e:
        return None


def process_village(village_cfg: dict,
                    output_paths: dict) -> dict:
    name        = village_cfg["name"]
    ori_path    = village_cfg["ori_path"]
    shp_path    = village_cfg["shp_path"]
    working_crs = village_cfg["working_crs"]
    split       = village_cfg["split"]

    log.info(f"\n{'='*55}")
    log.info(f"Processing: {name}  ({split})")
    log.info(f"{'='*55}")

    # Load ALL buildings from SHP
    buildings = load_buildings(shp_path, working_crs, name)
    if len(buildings) == 0:
        log.error(f"No buildings found, skipping {name}")
        return {}

    # ── KEY FIX: Filter buildings to ORI extent ───────────────
    with rasterio.open(ori_path) as src:
        ori_crs    = src.crs
        ori_bounds = box(*src.bounds)

    # Reproject buildings to ORI CRS for spatial filter
    buildings_ori = buildings.to_crs(ori_crs)
    within_mask   = buildings_ori.geometry.within(ori_bounds)
    buildings_ori = buildings_ori[within_mask].reset_index(drop=True)

    log.info(
        f"Buildings within ORI extent: "
        f"{len(buildings_ori)} / {len(buildings)}"
    )

    if len(buildings_ori) == 0:
        log.error(f"No buildings overlap with ORI for {name}")
        return {}

    # Log per-village breakdown
    if "Village_Na" in buildings_ori.columns:
        log.info("Villages within ORI:")
        for vname, cnt in buildings_ori["Village_Na"].value_counts().items():
            log.info(f"  {vname}: {cnt}")

    stats   = {cls: 0 for cls in CONFIG["class_dirs"]}
    skipped = 0

    with rasterio.open(ori_path) as src:
        for idx, row in tqdm(
            buildings_ori.iterrows(),
            total=len(buildings_ori),
            desc=f"  Cropping {name}",
        ):
            geom      = row.geometry
            roof_type = int(row["Roof_type"])

            crop = crop_building(src, geom, CONFIG["padding_px"])

            if crop is None:
                skipped += 1
                continue

            out_dir  = output_paths[roof_type]
            filename = f"{name}_{idx:05d}.tif"
            out_path = out_dir / filename

            crop_transform = rasterio.transform.from_bounds(
                geom.bounds[0] - CONFIG["padding_px"] * abs(src.transform.a),
                geom.bounds[1] - CONFIG["padding_px"] * abs(src.transform.a),
                geom.bounds[2] + CONFIG["padding_px"] * abs(src.transform.a),
                geom.bounds[3] + CONFIG["padding_px"] * abs(src.transform.a),
                crop.shape[2],
                crop.shape[1],
            )

            with rasterio.open(
                out_path, "w",
                driver="GTiff",
                height=crop.shape[1],
                width=crop.shape[2],
                count=crop.shape[0],
                dtype=crop.dtype,
                crs=src.crs,
                transform=crop_transform,
                compress="lzw",
            ) as dst:
                dst.write(crop)

            stats[roof_type] += 1

    total_saved = sum(stats.values())
    class_names = {1:"RCC", 2:"Tiled", 3:"Tin", 4:"Others"}
    log.info(f"\nResults for {name}:")
    log.info(f"  Buildings in ORI:  {len(buildings_ori)}")
    log.info(f"  Crops saved:       {total_saved}")
    log.info(f"  Skipped:           {skipped}")
    for rtype, count in stats.items():
        log.info(
            f"  {class_names[rtype]:8s}: {count:4d} crops"
        )

    return stats


def main() -> None:
    log.info("=" * 55)
    log.info("Roof Crop Extraction for EfficientNetV2-S")
    log.info("=" * 55)

    total_stats = {cls: 0 for cls in CONFIG["class_dirs"]}

    for village_cfg in CONFIG["villages"]:
        split       = village_cfg["split"]
        output_paths = setup_output_dirs(
            CONFIG["output_dir"], split
        )

        stats = process_village(village_cfg, output_paths)

        for cls, count in stats.items():
            total_stats[cls] += count

    # Final summary
    class_names = {1:"RCC", 2:"Tiled", 3:"Tin", 4:"Others"}
    log.info("\n" + "=" * 55)
    log.info("TOTAL CROPS GENERATED:")
    total = sum(total_stats.values())
    for rtype, count in total_stats.items():
        pct = count / total * 100 if total > 0 else 0
        log.info(
            f"  {class_names[rtype]:8s} (class {rtype}): "
            f"{count:5d}  ({pct:.1f}%)"
        )
    log.info(f"  {'TOTAL':8s}          : {total:5d}")
    log.info("\nNext: train EfficientNetV2-S on data/roof_crops/")


if __name__ == "__main__":
    main()