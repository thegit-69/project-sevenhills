"""
Test tiling script — single village (BADETUMNAR)
Changes from v1:
  - Removed resampling (Gemini was right, skip it)
  - Added windowed processing to handle large TIFs in 16GB RAM
  - Processes ORI in horizontal strips, not all at once

Run: python src/preprocessing/test_tiling.py
"""

import logging
import csv
from pathlib import Path
from typing import Optional
import numpy as np

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import mapping
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "village_name":   "BADETUMNAR",
    "state":          "CG",
    "split":          "train",

    # Update these to your actual WSL2 paths
    # Windows E:\ appears as /mnt/e/ in WSL2
    "ori_path": (
        "/home/kalki/projects/BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO.tif"
    ),
    "shp_dir":    "/home/kalki/projects/CG-shp-file",
    "output_dir": "/mnt/e/datasets/tiles",

    # CRS — reproject SHP to match ORI, no resampling
    "working_crs": "EPSG:32644",

    # Tile settings
    "tile_size": 512,
    "stride":    256,

    # Strip height for windowed processing
    # 3000 rows × 213734 cols × 3 bands × 1 byte ≈ 1.9 GB per strip
    # Safe for 16 GB RAM
    "strip_height": 3000,

    # Filtering
    "min_annotation_ratio": 0.02,
    "max_nodata_ratio":     0.30,
}

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── LAYER CONFIG ──────────────────────────────────────────────────────────────
# (filename, mask_value, buffer_metres or None)
# Last entry = highest priority (buildings override everything)
LAYERS = [
    ("Road.shp",              2, None),
    ("Road_Centre_Line.shp",  2, 3.0),
    ("Water_Body.shp",        3, None),
    ("Water_Body_Line.shp",   3, 4.0),
    ("Waterbody_Point.shp",   3, 5.0),
    ("Utility_Poly.shp",      4, None),
    ("Built_Up_Area_type.shp",1, None),  # burns last = highest priority
]


def get_reprojected_profile(ori_path: str,
                             working_crs: str) -> dict:
    """
    Calculates the output profile for reprojection
    WITHOUT resampling — keeps native pixel size.
    Returns rasterio profile dict.
    """
    with rasterio.open(ori_path) as src:
        # No resolution= argument = keeps native pixel size
        transform, width, height = calculate_default_transform(
            src.crs,
            working_crs,
            src.width,
            src.height,
            *src.bounds,
        )
        profile = src.profile.copy()
        profile.update({
            "crs":       working_crs,
            "transform": transform,
            "width":     width,
            "height":    height,
            "driver":    "GTiff",
        })
        log.info(
            f"ORI native pixel size: "
            f"{abs(src.transform.a):.4f}m/px"
        )
        log.info(
            f"Reprojected size: {width} x {height} px "
            f"in {working_crs} (no resampling)"
        )
        return profile


def load_and_reproject_vectors(shp_dir: str,
                                working_crs: str) -> dict:
    """
    Loads all shapefiles once, reprojects to working CRS,
    explodes multi-geometries, applies buffers.
    Returns dict of {layer_name: geodataframe}.
    Pre-loading saves time vs reloading per strip.
    """
    shp_path = Path(shp_dir)
    loaded = {}

    for fname, value, buffer_m in LAYERS:
        fpath = shp_path / fname
        if not fpath.exists():
            log.warning(f"Missing shapefile: {fname} — skipping")
            continue

        try:
            gdf = gpd.read_file(str(fpath))
            if len(gdf) == 0:
                log.warning(f"Empty: {fname}")
                continue

            # Reproject to working CRS
            gdf = gdf.to_crs(working_crs)

            # Explode multi-geometries
            gdf = gdf.explode(index_parts=False).reset_index(drop=True)

            # Filter Under Construction for buildings
            if "Built_Up_Area" in fname and "Remarks" in gdf.columns:
                before = len(gdf)
                uc = gdf["Remarks"].str.upper().str.contains(
                    "UNDER CONSTRUCTION", na=False
                )
                gdf = gdf[~uc].reset_index(drop=True)
                log.info(
                    f"Buildings: removed {before - len(gdf)} "
                    f"under-construction, kept {len(gdf)}"
                )

            # Apply buffer to lines and points
            if buffer_m is not None:
                gdf["geometry"] = gdf.geometry.buffer(buffer_m)

            # Drop invalid geometries
            gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]

            loaded[fname] = (gdf, value)
            log.info(f"Loaded {fname}: {len(gdf)} features, "
                     f"class {value}")

        except Exception as e:
            log.error(f"Failed to load {fname}: {e}")

    return loaded


def rasterize_strip(loaded_vectors: dict,
                    transform,
                    strip_height: int,
                    full_width: int) -> np.ndarray:
    """
    Rasterizes all vector layers into a mask strip.
    strip_transform: affine transform for this strip.
    Returns uint8 array of shape (strip_height, full_width).
    """
    mask = np.zeros((strip_height, full_width), dtype=np.uint8)

    for fname, (gdf, value) in loaded_vectors.items():
        if len(gdf) == 0:
            continue

        shapes = [
            (mapping(geom), value)
            for geom in gdf.geometry
            if geom is not None
        ]

        if not shapes:
            continue

        burned = rasterize(
            shapes=shapes,
            out_shape=(strip_height, full_width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False
        )

        # Higher priority layers override lower ones
        mask[burned > 0] = burned[burned > 0]

    return mask


def process_village_windowed(config: dict) -> list:
    """
    Main processing function using windowed (strip) approach.
    Processes the ORI in horizontal strips to stay within RAM.
    Returns manifest rows list.
    """
    village    = config["village_name"]
    working_crs = config["working_crs"]
    tile_size  = config["tile_size"]
    stride     = config["stride"]
    strip_h    = config["strip_height"]
    split      = config["split"]

    # ── Step 1: Get reprojected profile (no resampling) ──────────────────
    log.info("Calculating reprojection profile...")
    profile = get_reprojected_profile(
        config["ori_path"], working_crs
    )
    full_h = profile["height"]
    full_w = profile["width"]
    out_transform = profile["transform"]

    # ── Step 2: Load all vectors once ────────────────────────────────────
    log.info("Loading and reprojecting all shapefiles...")
    loaded_vectors = load_and_reproject_vectors(
        config["shp_dir"], working_crs
    )

    # ── Step 3: Setup output directories ─────────────────────────────────
    img_dir  = Path(config["output_dir"]) / split / village / "img"
    mask_dir = Path(config["output_dir"]) / split / village / "mask"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    total = saved = skipped_nodata = skipped_annotation = 0

    # ── Step 4: Process in strips ─────────────────────────────────────────
    # Each strip is strip_h rows tall
    # We tile within each strip
    strip_starts = list(range(0, full_h - tile_size + 1, stride))

    # Deduplicate strips — only start new strip when
    # we move past the previous strip boundary
    strip_boundaries = list(range(0, full_h, strip_h))
    if strip_boundaries[-1] < full_h:
        strip_boundaries.append(full_h)

    log.info(
        f"Processing {village}: {full_w}x{full_h}px "
        f"in {len(strip_boundaries)-1} strips of {strip_h} rows"
    )

    with rasterio.open(config["ori_path"]) as src:

        for strip_idx in range(len(strip_boundaries) - 1):
            strip_row_start = strip_boundaries[strip_idx]
            strip_row_end   = min(
                strip_boundaries[strip_idx + 1], full_h
            )
            actual_strip_h  = strip_row_end - strip_row_start

            log.info(
                f"Strip {strip_idx+1}/{len(strip_boundaries)-1}: "
                f"rows {strip_row_start}–{strip_row_end}"
            )

            # Calculate strip transform in working CRS
            from rasterio.transform import from_bounds
            import rasterio.transform as rt

            strip_top    = out_transform.f + strip_row_start * out_transform.e
            strip_bottom = out_transform.f + strip_row_end   * out_transform.e
            strip_left   = out_transform.c
            strip_right  = out_transform.c + full_w * out_transform.a

            strip_transform = from_bounds(
                strip_left, strip_bottom,
                strip_right, strip_top,
                full_w, actual_strip_h
            )

            # Read and reproject ORI strip
            from rasterio.warp import reproject as warp_reproject
            dst_strip = np.zeros(
                (src.count, actual_strip_h, full_w),
                dtype=np.uint8
            )

            for band in range(1, src.count + 1):
                warp_reproject(
                    source=rasterio.band(src, band),
                    destination=dst_strip[band - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=strip_transform,
                    dst_crs=working_crs,
                    resampling=Resampling.nearest,
                )

            # Build mask strip
            mask_strip = rasterize_strip(
                loaded_vectors,
                strip_transform,
                actual_strip_h,
                full_w
            )

            # Tile within this strip
            tile_rows_in_strip = [
                r for r in range(0, actual_strip_h - tile_size + 1, stride)
            ]

            for local_row in tqdm(
                tile_rows_in_strip,
                desc=f"  Tiling strip {strip_idx+1}",
                leave=False
            ):
                for col in range(0, full_w - tile_size + 1, stride):
                    total += 1

                    ori_tile  = dst_strip[
                        :,
                        local_row:local_row + tile_size,
                        col:col + tile_size
                    ]
                    mask_tile = mask_strip[
                        local_row:local_row + tile_size,
                        col:col + tile_size
                    ]

                    # Skip nodata
                    if np.mean(ori_tile < 5) > config["max_nodata_ratio"]:
                        skipped_nodata += 1
                        continue

                    # Skip unannotated
                    if np.mean(mask_tile > 0) < config["min_annotation_ratio"]:
                        skipped_annotation += 1
                        continue

                    # Global row for filename
                    global_row = strip_row_start + local_row

                    # Tile geotransform
                    tile_transform = from_bounds(
                        strip_left  + col * out_transform.a,
                        strip_top   + (local_row + tile_size) * out_transform.e,
                        strip_left  + (col + tile_size) * out_transform.a,
                        strip_top   + local_row * out_transform.e,
                        tile_size, tile_size
                    )

                    # Save image tile
                    img_path = img_dir / f"img_{global_row:06d}_{col:06d}.tif"
                    with rasterio.open(
                        img_path, "w",
                        driver="GTiff",
                        height=tile_size, width=tile_size,
                        count=src.count,
                        dtype=ori_tile.dtype,
                        crs=working_crs,
                        transform=tile_transform,
                        compress="lzw"
                    ) as dst:
                        dst.write(ori_tile)

                    # Save mask tile
                    mask_path = mask_dir / f"mask_{global_row:06d}_{col:06d}.tif"
                    with rasterio.open(
                        mask_path, "w",
                        driver="GTiff",
                        height=tile_size, width=tile_size,
                        count=1,
                        dtype=np.uint8,
                        crs=working_crs,
                        transform=tile_transform,
                        compress="lzw"
                    ) as dst:
                        dst.write(mask_tile[np.newaxis])

                    manifest_rows.append({
                        "img_path":  str(img_path),
                        "mask_path": str(mask_path),
                        "village":   village,
                        "state":     config["state"],
                        "split":     split,
                        "row":       global_row,
                        "col":       col,
                    })
                    saved += 1

    log.info(f"\n{'='*50}")
    log.info(f"Village: {village}")
    log.info(f"  Total tile positions:     {total:,}")
    log.info(f"  Skipped (nodata):         {skipped_nodata:,}")
    log.info(f"  Skipped (no annotation):  {skipped_annotation:,}")
    log.info(f"  Saved tiles:              {saved:,}")
    log.info(f"{'='*50}")

    return manifest_rows


def save_manifest(rows: list, output_dir: str, split: str) -> None:
    if not rows:
        log.warning("No tiles to save in manifest")
        return
    path = Path(output_dir) / f"{split}_manifest.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Manifest saved: {path}  ({len(rows)} tiles)")


def main() -> None:
    log.info("=" * 60)
    log.info(f"Test tiling: {CONFIG['village_name']}")
    log.info(f"No resampling — native pixel size preserved")
    log.info("=" * 60)

    rows = process_village_windowed(CONFIG)
    save_manifest(rows, CONFIG["output_dir"], CONFIG["split"])

    log.info("\nNext steps:")
    log.info("1. Open a tile pair in QGIS to verify alignment")
    log.info("2. Check mask has buildings/roads/water labeled")
    log.info("3. If good → run on all 10 villages on CPU VM")


if __name__ == "__main__":
    main()