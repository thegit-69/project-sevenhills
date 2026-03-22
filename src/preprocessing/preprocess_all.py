"""
Full preprocessing pipeline — all 10 training villages.
Generates 512x512 tiles for SegFormer + DeepLabV3+ training.

Usage:
    python src/preprocessing/preprocess_all.py

Output:
    data/tiles/train/{village}/img/
    data/tiles/train/{village}/mask/
    data/tiles/val/{village}/img/
    data/tiles/val/{village}/mask/
    data/tiles/train_manifest.csv
    data/tiles/val_manifest.csv
"""

import logging
import csv
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import mapping
from tqdm import tqdm

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── OUTPUT CONFIG ─────────────────────────────────────────────────────────────
OUTPUT_DIR   = "/home/kalki/data/tiles"    # update if needed
TILE_SIZE    = 512
STRIDE       = 256
STRIP_HEIGHT = 3000   # rows per strip — safe for 32GB RAM on VM
MIN_ANNO     = 0.02   # skip tiles with <2% labeled pixels
MAX_NODATA   = 0.30   # skip tiles with >30% black pixels

# ── VILLAGE CONFIG ────────────────────────────────────────────────────────────
# Update base paths to match where data lives on your VM
CG_BASE  = "/home/kalki/data/raw/CG"
CG_SHP   = "/home/kalki/data/raw/CG/shp-file"
PB_BASE  = "/home/kalki/data/raw/PB"
PB_SHP   = "/home/kalki/data/raw/PB/shp-file"

VILLAGES = [
    # ── CG Training villages ──────────────────────────────────────────────
   ''' {
        "name":        "BADETUMNAR",
        "state":       "CG",
        "split":       "train",
        "ori":         f"{CG_BASE}/BADETUMNAR_450157_BANGAPAL_450155_CHHOTETUMAR_450149_MOFALNAR_450150_ORTHO.tif",
        "shp_dir":     CG_SHP,
        "working_crs": "EPSG:32644",
        "shp_prefix":  "CG",   # tells script which SHP filenames to use
    },'''
    {
        "name":        "MURDANDA",
        "state":       "CG",
        "split":       "train",
        "ori":         f"{CG_BASE}/MURDANDA_450879_AWAPALLI_CHINTAKONTA_ORTHO.tif",
        "shp_dir":     CG_SHP,
        "working_crs": "EPSG:32644",
        "shp_prefix":  "CG",
    },
    {
        "name":        "NAGUL",
        "state":       "CG",
        "split":       "train",
        "ori":         f"{CG_BASE}/NAGUL_450171_MADASE_450172_GHOTPAL_450137_ORTHO.tif",
        "shp_dir":     CG_SHP,
        "working_crs": "EPSG:32644",
        "shp_prefix":  "CG",
    },
    '''{
        "name":        "SAMLUR",
        "state":       "CG",
        "split":       "train",
        "ori":         f"{CG_BASE}/SAMLUR_450163_SIYANAR_450164_KUTULNAR_450165_BINJAM_450166_JHODIYAWADAM_450167_ORTHO.tif",
        "shp_dir":     CG_SHP,
        "working_crs": "EPSG:32644",
        "shp_prefix":  "CG",
    },
    {
        "name":        "KUTRU",
        "state":       "CG",
        "split":       "val",
        # Use converted TIF not ECW
        "ori":         f"{CG_BASE}/KUTRU_Converted.tif",
        "shp_dir":     CG_SHP,
        "working_crs": "EPSG:3857",   # matches ORI
        "shp_prefix":  "CG",
    },

    # ── PB Training villages ──────────────────────────────────────────────
    {
        "name":        "NADALA",
        "state":       "PB",
        "split":       "train",
        "ori":         f"{PB_BASE}/28996_NADALA_ORTHO.tif",
        "shp_dir":     PB_SHP,
        "working_crs": "EPSG:32643",
        "shp_prefix":  "PB",
    },'''
    {
        "name":        "PINDORI",
        "state":       "PB",
        "split":       "train",
        "ori":         f"{PB_BASE}/PINDORI_MAYA_SINGH_TUGALWAL_28456_ortho.tif",
        "shp_dir":     PB_SHP,
        "working_crs": "EPSG:32643",
        "shp_prefix":  "PB",
    },
    '''{
        "name":        "TIMMOWAL",
        "state":       "PB",
        "split":       "train",
        "ori":         f"{PB_BASE}/TIMMOWAL_37695_ORI.tif",
        "shp_dir":     PB_SHP,
        "working_crs": "EPSG:32643",
        "shp_prefix":  "PB",
    },
    {
        "name":        "FATTU",
        "state":       "PB",
        "split":       "train",
        "ori":         f"{PB_BASE}/37458_fattu_bhila_ortho_3857.tif",
        "shp_dir":     PB_SHP,
        "working_crs": "EPSG:3857",   # matches ORI
        "shp_prefix":  "PB",
    },
    {
        "name":        "BAGGA",
        "state":       "PB",
        "split":       "val",
        "ori":         f"{PB_BASE}/37774_bagga_ortho_3857.tif",
        "shp_dir":     PB_SHP,
        "working_crs": "EPSG:3857",   # matches ORI
        "shp_prefix":  "PB",
    },'''
]

# ── LAYER DEFINITIONS PER STATE ───────────────────────────────────────────────
# CG uses Built_Up_Area_type.shp
# PB uses Built_Up_Area_typ.shp  (no trailing 'e')
# PB uses Utility_Poly_.shp      (trailing underscore)
# (filename, mask_value, buffer_metres or None)
# Last entry = highest priority

CG_LAYERS = [
    ("Road.shp",               2, None),
    ("Road_Centre_Line.shp",   2, 3.0),
    ("Water_Body.shp",         3, None),
    ("Water_Body_Line.shp",    3, 4.0),
    ("Waterbody_Point.shp",    3, 8.0),
    ("Utility_Poly.shp",       4, None),
    ("Built_Up_Area_type.shp", 1, None),  # highest priority
]

PB_LAYERS = [
    ("Road.shp",               2, None),
    ("Road_Centre_Line.shp",   2, 3.0),
    ("Water_Body.shp",         3, None),
    ("Water_Body_Line.shp",    3, 4.0),
    ("Waterbody_Point.shp",    3, 8.0),
    ("Utility_Poly_.shp",      4, None),  # trailing underscore
    ("Built_Up_Area_typ.shp",  1, None),  # no trailing 'e'
]


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def get_layers_for_state(shp_prefix: str) -> list:
    return CG_LAYERS if shp_prefix == "CG" else PB_LAYERS


def load_vectors(shp_dir: str, working_crs: str,
                 shp_prefix: str) -> dict:
    """
    Loads all shapefiles for this village.
    Reprojects, explodes multi-geometries, applies buffers.
    Returns dict of {filename: (geodataframe, mask_value)}
    """
    layers  = get_layers_for_state(shp_prefix)
    shp_path = Path(shp_dir)
    loaded  = {}

    for fname, value, buffer_m in layers:
        fpath = shp_path / fname
        if not fpath.exists():
            log.warning(f"  Missing: {fname} — skipping layer")
            continue

        try:
            gdf = gpd.read_file(str(fpath))
            if len(gdf) == 0:
                log.warning(f"  Empty: {fname}")
                continue

            # Reproject to working CRS
            gdf = gdf.to_crs(working_crs)

            # Explode ALL multi-geometries
            # Handles MultiPolygon, MultiLineString, MultiPoint
            gdf = gdf.explode(index_parts=False).reset_index(drop=True)

            # Filter Under Construction buildings
            if "Built_Up_Area" in fname and "Remarks" in gdf.columns:
                before = len(gdf)
                uc = gdf["Remarks"].str.upper().str.contains(
                    "UNDER CONSTRUCTION", na=False
                )
                gdf = gdf[~uc].reset_index(drop=True)
                removed = before - len(gdf)
                if removed:
                    log.info(f"  {fname}: removed {removed} under-construction")

            # Buffer lines and points
            if buffer_m is not None:
                gdf["geometry"] = gdf.geometry.buffer(buffer_m)

            # Drop invalid geometries
            gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]

            if len(gdf) == 0:
                log.warning(f"  {fname}: no valid geometries after processing")
                continue

            loaded[fname] = (gdf, value)
            log.info(f"  Loaded {fname}: {len(gdf)} features → class {value}")

        except Exception as e:
            log.error(f"  Failed {fname}: {e}")

    return loaded


def rasterize_strip(loaded_vectors: dict, transform,
                    strip_h: int, full_w: int) -> np.ndarray:
    """Burns all vector layers into a single mask strip."""
    mask = np.zeros((strip_h, full_w), dtype=np.uint8)

    for fname, (gdf, value) in loaded_vectors.items():
        shapes = [
            (mapping(geom), value)
            for geom in gdf.geometry
            if geom is not None
        ]
        if not shapes:
            continue

        burned = rasterize(
            shapes=shapes,
            out_shape=(strip_h, full_w),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False
        )
        # Higher priority layers (burned last) override lower ones
        mask[burned > 0] = burned[burned > 0]

    return mask


def process_village(vcfg: dict, all_manifests: dict) -> None:
    """
    Processes one village end-to-end.
    Appends tile info to all_manifests[split].
    """
    name        = vcfg["name"]
    state       = vcfg["state"]
    split       = vcfg["split"]
    ori_path    = vcfg["ori"]
    shp_dir     = vcfg["shp_dir"]
    working_crs = vcfg["working_crs"]
    shp_prefix  = vcfg["shp_prefix"]

    log.info(f"\n{'='*60}")
    log.info(f"Village: {name}  |  State: {state}  |  Split: {split}")
    log.info(f"Working CRS: {working_crs}")
    log.info(f"{'='*60}")

    # ── Get output dimensions without resampling ──────────────────────────
    with rasterio.open(ori_path) as src:
        transform, full_w, full_h = calculate_default_transform(
            src.crs, working_crs,
            src.width, src.height,
            *src.bounds,
        )
        band_count = src.count
        dtype      = src.meta["dtype"]
        log.info(
            f"ORI: {src.width}x{src.height}px  "
            f"@ {abs(src.transform.a):.4f}m/px  "
            f"native CRS: {src.crs}"
        )
        log.info(
            f"Output: {full_w}x{full_h}px in {working_crs}"
        )

    # ── Load all shapefiles ───────────────────────────────────────────────
    log.info("Loading shapefiles...")
    loaded_vectors = load_vectors(shp_dir, working_crs, shp_prefix)

    if not loaded_vectors:
        log.error(f"No shapefiles loaded for {name} — skipping")
        return

    # ── Setup output dirs ─────────────────────────────────────────────────
    img_dir  = Path(OUTPUT_DIR) / split / name / "img"
    mask_dir = Path(OUTPUT_DIR) / split / name / "mask"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # ── Strip boundaries ──────────────────────────────────────────────────
    strip_bounds = list(range(0, full_h, STRIP_HEIGHT))
    if strip_bounds[-1] < full_h:
        strip_bounds.append(full_h)

    log.info(
        f"Processing in {len(strip_bounds)-1} strips "
        f"of {STRIP_HEIGHT} rows each"
    )

    total = saved = skipped_nd = skipped_ann = 0

    with rasterio.open(ori_path) as src:

        for s_idx in range(len(strip_bounds) - 1):
            row_start  = strip_bounds[s_idx]
            row_end    = strip_bounds[s_idx + 1]
            actual_h   = row_end - row_start

            # Strip geographic bounds
            strip_top    = transform.f + row_start * transform.e
            strip_bottom = transform.f + row_end   * transform.e
            strip_left   = transform.c
            strip_right  = transform.c + full_w * transform.a

            strip_tf = from_bounds(
                strip_left, strip_bottom,
                strip_right, strip_top,
                full_w, actual_h
            )

            log.info(
                f"  Strip {s_idx+1}/{len(strip_bounds)-1}: "
                f"rows {row_start}–{row_end}"
            )

            # Reproject ORI strip into memory
            dst_strip = np.zeros(
                (band_count, actual_h, full_w),
                dtype=np.uint8
            )
            for band in range(1, band_count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=dst_strip[band - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=strip_tf,
                    dst_crs=working_crs,
                    resampling=Resampling.nearest,
                )

            # Build mask strip
            mask_strip = rasterize_strip(
                loaded_vectors, strip_tf, actual_h, full_w
            )

            # Tile within strip
            tile_rows = range(0, actual_h - TILE_SIZE + 1, STRIDE)

            for local_row in tqdm(
                tile_rows,
                desc=f"    Tiling strip {s_idx+1}",
                leave=False
            ):
                for col in range(0, full_w - TILE_SIZE + 1, STRIDE):
                    total += 1

                    img_tile  = dst_strip[
                        :,
                        local_row:local_row + TILE_SIZE,
                        col:col + TILE_SIZE
                    ]
                    mask_tile = mask_strip[
                        local_row:local_row + TILE_SIZE,
                        col:col + TILE_SIZE
                    ]

                    # Skip nodata tiles
                    if np.mean(img_tile < 5) > MAX_NODATA:
                        skipped_nd += 1
                        continue

                    # Skip unannotated tiles
                    if np.mean(mask_tile > 0) < MIN_ANNO:
                        skipped_ann += 1
                        continue

                    global_row = row_start + local_row

                    # Tile transform
                    tile_tf = from_bounds(
                        strip_left + col * transform.a,
                        strip_top  + (local_row + TILE_SIZE) * transform.e,
                        strip_left + (col + TILE_SIZE) * transform.a,
                        strip_top  + local_row * transform.e,
                        TILE_SIZE, TILE_SIZE
                    )

                    # Save image tile
                    img_p = img_dir / f"img_{global_row:06d}_{col:06d}.tif"
                    with rasterio.open(
                        img_p, "w", driver="GTiff",
                        height=TILE_SIZE, width=TILE_SIZE,
                        count=band_count, dtype=np.uint8,
                        crs=working_crs, transform=tile_tf,
                        compress="lzw"
                    ) as dst:
                        dst.write(img_tile)

                    # Save mask tile
                    mask_p = mask_dir / f"mask_{global_row:06d}_{col:06d}.tif"
                    with rasterio.open(
                        mask_p, "w", driver="GTiff",
                        height=TILE_SIZE, width=TILE_SIZE,
                        count=1, dtype=np.uint8,
                        crs=working_crs, transform=tile_tf,
                        compress="lzw"
                    ) as dst:
                        dst.write(mask_tile[np.newaxis])

                    all_manifests[split].append({
                        "img_path":  str(img_p),
                        "mask_path": str(mask_p),
                        "village":   name,
                        "state":     state,
                        "split":     split,
                    })
                    saved += 1

    # Village summary
    log.info(f"\n  Results — {name}:")
    log.info(f"    Positions checked:  {total:,}")
    log.info(f"    Skipped (nodata):   {skipped_nd:,}")
    log.info(f"    Skipped (no label): {skipped_ann:,}")
    log.info(f"    Tiles saved:        {saved:,}")

    # Class distribution of saved masks
    sample_masks = list(mask_dir.glob("*.tif"))[:50]
    class_pixels = {0:0, 1:0, 2:0, 3:0, 4:0}
    for mp in sample_masks:
        with rasterio.open(mp) as ms:
            data = ms.read(1)
            for cls in class_pixels:
                class_pixels[cls] += int(np.sum(data == cls))
    total_px = sum(class_pixels.values()) or 1
    names = {0:"bg", 1:"building", 2:"road", 3:"water", 4:"utility"}
    log.info(f"    Class distribution (sample of 50 masks):")
    for cls, px in class_pixels.items():
        log.info(f"      {names[cls]:10s}: {px/total_px*100:.1f}%")


def save_manifests(manifests: dict) -> None:
    """Saves train and val manifest CSVs."""
    for split, rows in manifests.items():
        if not rows:
            log.warning(f"No tiles for split: {split}")
            continue
        out = Path(OUTPUT_DIR) / f"{split}_manifest.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Manifest: {out}  ({len(rows)} tiles)")


def main() -> None:
    log.info("=" * 60)
    log.info("Full preprocessing pipeline — all 10 villages")
    log.info("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Accumulate manifests across all villages
    manifests = {"train": [], "val": []}

    for vcfg in VILLAGES:
        process_village(vcfg, manifests)

    # Save combined manifests
    save_manifests(manifests)

    # Final summary
    log.info("\n" + "=" * 60)
    log.info("ALL VILLAGES COMPLETE")
    log.info(f"  Train tiles: {len(manifests['train']):,}")
    log.info(f"  Val tiles:   {len(manifests['val']):,}")
    log.info(f"  Output dir:  {OUTPUT_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
