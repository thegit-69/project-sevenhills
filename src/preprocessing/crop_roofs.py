"""
Roof crop extraction for EfficientNetV2-S classifier.
Reads Built_Up_Area shapefile, crops each building from ORI,
saves to class folders: 1=RCC, 2=Tiled, 3=Tin, 4=Others

Output structure:
  /home/kalki/data/roof_crops/train/1_RCC/BADETUMNAR_00001.tif
  /home/kalki/data/roof_crops/train/2_Tiled/BADETUMNAR_00002.tif
  /home/kalki/data/roof_crops/train/3_Tin/BADETUMNAR_00003.tif
  /home/kalki/data/roof_crops/train/4_Others/BADETUMNAR_00004.tif
  /home/kalki/data/roof_crops/val/...

Run: python src/preprocessing/crop_roofs.py
"""

import logging
from pathlib import Path
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from shapely.geometry import mapping, box
from tqdm import tqdm

# ── PATHS ─────────────────────────────────────────────────────────────────────
CG_BASE  = "/home/kalki/data/raw/CG"
CG_SHP   = f"{CG_BASE}/shp-file/Built_Up_Area_type.shp"
PB_BASE  = "/home/kalki/data/raw/PB"
PB_SHP   = f"{PB_BASE}/shp-file/Built_Up_Area_typ.shp"
OUT_DIR  = "/home/kalki/data/roof_crops"

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "villages": [
        # ── CG Training villages ──────────────────────────────────────────
        {
            "name":        "BADETUMNAR",
            "state":       "CG",
            "split":       "train",
            "ori_path":    f"{CG_BASE}/BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO.tif",
            "shp_path":    CG_SHP,
            "working_crs": "EPSG:32644",
        },
        {
            "name":        "MURDANDA",
            "state":       "CG",
            "split":       "train",
            "ori_path":    f"{CG_BASE}/MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO.tif",
            "shp_path":    CG_SHP,
            "working_crs": "EPSG:32644",
        },
        {
            "name":        "NAGUL",
            "state":       "CG",
            "split":       "train",
            "ori_path":    f"{CG_BASE}/NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO.tif",
            "shp_path":    CG_SHP,
            "working_crs": "EPSG:32644",
        },
        {
            "name":        "SAMLUR",
            "state":       "CG",
            "split":       "train",
            "ori_path":    f"{CG_BASE}/SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO.tif",
            "shp_path":    CG_SHP,
            "working_crs": "EPSG:32644",
        },
        {
            "name":        "KUTRU",
            "state":       "CG",
            "split":       "val",
            "ori_path":    f"{CG_BASE}/KUTRU_Converted.tif",
            "shp_path":    CG_SHP,
            "working_crs": "EPSG:3857",   # ORI is 3857 — fixed
        },

        # ── PB Training villages ──────────────────────────────────────────
        {
            "name":        "NADALA",
            "state":       "PB",
            "split":       "train",
            "ori_path":    f"{PB_BASE}/28996_NADALA_ORTHO.tif",
            "shp_path":    PB_SHP,
            "working_crs": "EPSG:32643",
        },
        {
            "name":        "PINDORI",
            "state":       "PB",
            "split":       "train",
            "ori_path":    f"{PB_BASE}/PINDORI_MAYA_SINGH_TUGALWAL_28456_ortho.tif",
            "shp_path":    PB_SHP,
            "working_crs": "EPSG:32643",
        },
        {
            "name":        "TIMMOWAL",
            "state":       "PB",
            "split":       "train",
            "ori_path":    f"{PB_BASE}/TIMMOWAL_37695_ORI.tif",
            "shp_path":    PB_SHP,
            "working_crs": "EPSG:32643",
        },
        {
            "name":        "FATTU",
            "state":       "PB",
            "split":       "train",
            "ori_path":    f"{PB_BASE}/37458_fattu_bhila_ortho_3857.tif",
            "shp_path":    PB_SHP,
            "working_crs": "EPSG:3857",   # ORI is 3857 — fixed
        },
        {
            "name":        "BAGGA",
            "state":       "PB",
            "split":       "val",
            "ori_path":    f"{PB_BASE}/37774_bagga_ortho_3857.tif",
            "shp_path":    PB_SHP,
            "working_crs": "EPSG:3857",   # ORI is 3857 — fixed
        },
    ],

    "output_dir":      OUT_DIR,
    "padding_px":      10,
    "min_area_sqm":    8.0,
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
    paths = {}
    for val, dirname in CONFIG["class_dirs"].items():
        p = Path(output_dir) / split / dirname
        p.mkdir(parents=True, exist_ok=True)
        paths[val] = p
    return paths


def load_buildings(shp_path: str, working_crs: str) -> gpd.GeoDataFrame:
    """
    Loads building shapefile, reprojects, validates,
    removes under-construction and tiny buildings.
    """
    gdf = gpd.read_file(shp_path)
    log.info(f"  SHP total features: {len(gdf)}")

    # Reproject to working CRS
    gdf = gdf.to_crs(working_crs)

    # Explode MultiPolygons
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    # Remove under construction
    if "Remarks" in gdf.columns:
        before = len(gdf)
        uc = gdf["Remarks"].str.upper().str.contains(
            "UNDER CONSTRUCTION", na=False
        )
        gdf = gdf[~uc].reset_index(drop=True)
        if before - len(gdf) > 0:
            log.info(f"  Removed {before - len(gdf)} under-construction")

    # Keep only valid roof types
    if "Roof_type" not in gdf.columns:
        log.error("  Roof_type column not found!")
        return gpd.GeoDataFrame()

    gdf = gdf[gdf["Roof_type"].isin([1, 2, 3, 4])].reset_index(drop=True)

    # Remove tiny buildings
    if "Area_Sqm" in gdf.columns:
        before = len(gdf)
        gdf = gdf[gdf["Area_Sqm"] >= CONFIG["min_area_sqm"]].reset_index(drop=True)
        if before - len(gdf) > 0:
            log.info(f"  Removed {before - len(gdf)} tiny buildings (<{CONFIG['min_area_sqm']}m²)")

    # Remove invalid geometries
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]

    return gdf


def crop_building(src: rasterio.DatasetReader,
                  geom, padding_px: int) -> np.ndarray | None:
    """Crops one building from ORI with padding. Returns array or None."""
    try:
        px_size   = abs(src.transform.a)
        padding_m = padding_px * px_size
        bounds    = geom.bounds
        padded    = box(
            bounds[0] - padding_m,
            bounds[1] - padding_m,
            bounds[2] + padding_m,
            bounds[3] + padding_m,
        )

        out_image, _ = rio_mask(
            src, [mapping(padded)],
            crop=True, nodata=0, filled=True,
        )

        if out_image is None:
            return None
        if out_image.shape[1] < 4 or out_image.shape[2] < 4:
            return None
        if np.mean(out_image) < 5:
            return None

        return out_image

    except Exception:
        return None


def process_village(village_cfg: dict, output_paths: dict) -> dict:
    name        = village_cfg["name"]
    ori_path    = village_cfg["ori_path"]
    shp_path    = village_cfg["shp_path"]
    working_crs = village_cfg["working_crs"]
    split       = village_cfg["split"]

    log.info(f"\n{'='*55}")
    log.info(f"Village: {name}  |  State: {village_cfg['state']}  |  Split: {split}")
    log.info(f"Working CRS: {working_crs}")
    log.info(f"{'='*55}")

    # Load all buildings from SHP
    buildings = load_buildings(shp_path, working_crs)
    if len(buildings) == 0:
        log.error(f"No buildings loaded for {name}")
        return {}

    # Spatial filter — keep only buildings within this ORI
    with rasterio.open(ori_path) as src:
        ori_crs    = src.crs
        ori_bounds = box(*src.bounds)

    buildings_ori = buildings.to_crs(ori_crs)
    within        = buildings_ori.geometry.within(ori_bounds)
    buildings_ori = buildings_ori[within].reset_index(drop=True)

    log.info(f"  Buildings within ORI: {len(buildings_ori)} / {len(buildings)}")

    if len(buildings_ori) == 0:
        log.error(f"  No buildings overlap with ORI for {name}")
        return {}

    # Log roof type distribution for this village
    counts     = buildings_ori["Roof_type"].value_counts().sort_index()
    class_names = {1:"RCC", 2:"Tiled", 3:"Tin", 4:"Others"}
    for rtype, cnt in counts.items():
        pct = cnt / len(buildings_ori) * 100
        log.info(f"  Roof {rtype} ({class_names.get(rtype,'?')}): {cnt} ({pct:.1f}%)")

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

            # Build geotransform for crop
            px   = abs(src.transform.a)
            pad  = CONFIG["padding_px"] * px
            b    = geom.bounds
            tf   = rasterio.transform.from_bounds(
                b[0]-pad, b[1]-pad, b[2]+pad, b[3]+pad,
                crop.shape[2], crop.shape[1],
            )

            out_path = output_paths[roof_type] / f"{name}_{idx:05d}.tif"
            with rasterio.open(
                out_path, "w",
                driver="GTiff",
                height=crop.shape[1],
                width=crop.shape[2],
                count=crop.shape[0],
                dtype=crop.dtype,
                crs=src.crs,
                transform=tf,
                compress="lzw",
            ) as dst:
                dst.write(crop)

            stats[roof_type] += 1

    total = sum(stats.values())
    log.info(f"\n  Results — {name}:")
    log.info(f"    Buildings in ORI: {len(buildings_ori)}")
    log.info(f"    Crops saved:      {total}")
    log.info(f"    Skipped:          {skipped}")
    for rtype, cnt in stats.items():
        log.info(f"    {class_names[rtype]:8s}: {cnt:4d}")

    return stats


def main() -> None:
    log.info("=" * 55)
    log.info("Roof Crop Extraction — EfficientNetV2-S")
    log.info("=" * 55)

    total_stats = {cls: 0 for cls in CONFIG["class_dirs"]}

    for vcfg in CONFIG["villages"]:
        output_paths = setup_output_dirs(CONFIG["output_dir"], vcfg["split"])
        stats        = process_village(vcfg, output_paths)
        for cls, cnt in stats.items():
            total_stats[cls] += cnt

    class_names = {1:"RCC", 2:"Tiled", 3:"Tin", 4:"Others"}
    total       = sum(total_stats.values())
    log.info("\n" + "=" * 55)
    log.info("TOTAL CROPS GENERATED:")
    for rtype, cnt in total_stats.items():
        pct = cnt / total * 100 if total > 0 else 0
        log.info(f"  {class_names[rtype]:8s} (class {rtype}): {cnt:5d}  ({pct:.1f}%)")
    log.info(f"  {'TOTAL':8s}              : {total:5d}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()