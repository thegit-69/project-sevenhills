"""
Inference pipeline — SVAMITVA Feature Extraction
Runs SegFormer (buildings) + DeepLabV3+ (roads/water/utilities)
on test village orthophotos and produces OGC GeoPackage output.

Output GeoPackage layers:
  buildings  — polygons with roof_type from EfficientNetV2
  roads      — polygons from DeepLabV3+
  water      — polygons from DeepLabV3+
  utilities  — polygons from DeepLabV3+

Run: python src/inference/predict.py
"""

import logging
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.features import shapes
from rasterio.crs import CRS
import geopandas as gpd
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import pandas as pd
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CFG = {
    # Model paths
    "segformer_path":    "/home/kalki/models/segformer_binary/best_segformer.pth",
    "deeplabv3_path":    "/home/kalki/models/deeplabv3/best_deeplabv3.pth",
    "efficientnet_path": "/home/kalki/models/efficientnet/best_efficientnet.pth",

    # Output directory
    "output_dir": "/home/kalki/data/predictions",

    # Tiling
    "tile_size":    512,
    "stride":       256,
    "strip_height": 3000,
    "batch_size":   16,

    # Polygon simplification tolerance in metres
    "simplify_tolerance": 0.5,

    # Minimum area to keep (filter noise polygons)
    "min_building_area_m2": 8.0,
    "min_road_area_m2":     5.0,
    "min_water_area_m2":    10.0,
    "min_utility_area_m2":  2.0,

    # EfficientNet crop padding
    "crop_padding_px": 10,

    # Roof type class names
    "roof_classes": {0: "RCC", 1: "Tiled", 2: "Tin", 3: "Others"},
}

# ── TEST VILLAGES ─────────────────────────────────────────────────────────────
# Add all 10 test villages here
TEST_VILLAGES = [
    # CG test villages
    {
        "name":        "DIWANA_BARNALA",
        "ori_path":    "/home/kalki/data/raw/test/PB/live_demo_2/DIWANA_BARNALA_40082_ORTHO/DIWANA_BARNALA_40082_ORTHO.tif",
        "working_crs": "EPSG:32644",
    },
    # Add remaining 9 villages — update paths to match your actual filenames
    # {
    #     "name":        "KARTARPUR_AMRITSAR",
    #     "ori_path":    "/home/kalki/data/raw/test/...",
    #     "working_crs": "EPSG:32644",
    # },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
VAL_TRANSFORM = A.Compose([
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])

ROOF_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_segformer(path: str, device):
    log.info(f"Loading SegFormer from {path}")
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    cfg   = ckpt["cfg"]
    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg["model_name"],
        use_safetensors=True,
        num_labels=cfg["num_classes"],
        ignore_mismatched_sizes=True,
        id2label={str(i): n for i, n in enumerate(cfg["class_names"])},
        label2id={n: str(i) for i, n in enumerate(cfg["class_names"])},
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    log.info(f"SegFormer loaded — val mIoU was {ckpt['val_miou']:.4f}")
    return model, cfg


def load_deeplabv3(path: str, device):
    log.info(f"Loading DeepLabV3+ from {path}")
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    cfg   = ckpt["cfg"]
    model = smp.DeepLabV3Plus(
        encoder_name=cfg["encoder"],
        encoder_weights=None,
        in_channels=3,
        classes=cfg["num_classes"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    log.info(f"DeepLabV3+ loaded — val mIoU was {ckpt['val_miou']:.4f}")
    return model, cfg


def load_efficientnet(path: str, device):
    log.info(f"Loading EfficientNetV2 from {path}")
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    cfg   = ckpt["cfg"]
    model = timm.create_model(
        cfg["model_name"],
        pretrained=False,
        num_classes=cfg["num_classes"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    log.info(
        f"EfficientNetV2 loaded — val acc was {ckpt['val_acc']:.4f}"
    )
    return model, cfg


# ── INFERENCE HELPERS ─────────────────────────────────────────────────────────
def preprocess_tile(tile: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 tile to normalized CHW tensor."""
    if tile.shape[2] == 1:
        tile = np.repeat(tile, 3, axis=2)
    elif tile.shape[2] > 3:
        tile = tile[:, :, :3]
    aug = VAL_TRANSFORM(image=tile.astype(np.uint8))
    return aug["image"].unsqueeze(0)


def predict_full_image_segformer(
    model, ori_path: str, working_crs: str, device
) -> tuple[np.ndarray, object, object]:
    """
    Runs SegFormer on full orthophoto using windowed strips.
    Returns (prediction_array, transform, crs).
    Binary output: 0=background, 1=building
    """
    with rasterio.open(ori_path) as src:
        transform, full_w, full_h = calculate_default_transform(
            src.crs, working_crs,
            src.width, src.height, *src.bounds,
        )
        band_count = min(src.count, 3)

    tile_size   = CFG["tile_size"]
    stride      = CFG["stride"]
    strip_h     = CFG["strip_height"]

    # Accumulate softmax probabilities with Gaussian weighting
    prob_sum    = np.zeros((2, full_h, full_w), dtype=np.float16)
    weight_sum  = np.zeros((full_h, full_w),    dtype=np.float16)

    # Gaussian weight map — center pixels get higher weight
    cy, cx     = tile_size // 2, tile_size // 2
    y_idx, x_idx = np.mgrid[:tile_size, :tile_size]
    sigma      = tile_size * 0.25
    gauss      = np.exp(
        -((y_idx - cy)**2 + (x_idx - cx)**2) / (2 * sigma**2)
    ).astype(np.float16)

    strip_bounds = list(range(0, full_h, strip_h))
    if strip_bounds[-1] < full_h:
        strip_bounds.append(full_h)

    with rasterio.open(ori_path) as src:
        for s_idx in tqdm(
            range(len(strip_bounds) - 1),
            desc="  SegFormer strips", leave=False
        ):
            row_start = strip_bounds[s_idx]
            row_end   = strip_bounds[s_idx + 1]
            actual_h  = row_end - row_start

            strip_top    = transform.f + row_start * transform.e
            strip_bottom = transform.f + row_end   * transform.e
            strip_left   = transform.c
            strip_right  = transform.c + full_w * transform.a

            strip_tf = from_bounds(
                strip_left, strip_bottom,
                strip_right, strip_top,
                full_w, actual_h
            )

            dst_strip = np.zeros(
                (band_count, actual_h, full_w), dtype=np.uint8
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

            strip_img = np.transpose(dst_strip, (1, 2, 0))

            for local_row in range(
                0, actual_h - tile_size + 1, stride
            ):
                for col in range(0, full_w - tile_size + 1, stride):
                    tile = strip_img[
                        local_row:local_row + tile_size,
                        col:col + tile_size
                    ]
                    if np.mean(tile) < 5:
                        continue

                    inp = preprocess_tile(tile).to(device)
                    with torch.no_grad():
                        out    = model(pixel_values=inp)
                        logits = F.interpolate(
                            out.logits,
                            size=(tile_size, tile_size),
                            mode="bilinear", align_corners=False,
                        )
                        probs = torch.softmax(logits, dim=1)
                        probs = probs.squeeze(0).cpu().numpy()

                    gr = row_start + local_row
                    prob_sum[:, gr:gr+tile_size, col:col+tile_size] \
                        += probs * gauss
                    weight_sum[gr:gr+tile_size, col:col+tile_size] \
                        += gauss

    weight_sum = np.maximum(weight_sum, 1e-8)
    for c in range(2):
        prob_sum[c] /= weight_sum

    pred = np.argmax(prob_sum, axis=0).astype(np.uint8)
    return pred, transform, CRS.from_epsg(
        int(working_crs.split(":")[1])
    )


def predict_full_image_deeplabv3(
    model, ori_path: str, working_crs: str, device
) -> np.ndarray:
    """
    Runs DeepLabV3+ on full orthophoto.
    Returns prediction array:
      0=background, 1=road, 2=water, 3=utility
    """
    with rasterio.open(ori_path) as src:
        transform, full_w, full_h = calculate_default_transform(
            src.crs, working_crs,
            src.width, src.height, *src.bounds,
        )
        band_count = min(src.count, 3)

    tile_size  = CFG["tile_size"]
    stride     = CFG["stride"]
    strip_h    = CFG["strip_height"]

    prob_sum   = np.zeros((4, full_h, full_w), dtype=np.float16)
    weight_sum = np.zeros((full_h, full_w),    dtype=np.float16)

    cy, cx     = tile_size // 2, tile_size // 2
    y_idx, x_idx = np.mgrid[:tile_size, :tile_size]
    sigma      = tile_size * 0.25
    gauss      = np.exp(
        -((y_idx - cy)**2 + (x_idx - cx)**2) / (2 * sigma**2)
    ).astype(np.float16)

    strip_bounds = list(range(0, full_h, strip_h))
    if strip_bounds[-1] < full_h:
        strip_bounds.append(full_h)

    with rasterio.open(ori_path) as src:
        for s_idx in tqdm(
            range(len(strip_bounds) - 1),
            desc="  DeepLabV3+ strips", leave=False
        ):
            row_start = strip_bounds[s_idx]
            row_end   = strip_bounds[s_idx + 1]
            actual_h  = row_end - row_start

            strip_top    = transform.f + row_start * transform.e
            strip_bottom = transform.f + row_end   * transform.e
            strip_left   = transform.c
            strip_right  = transform.c + full_w * transform.a

            strip_tf = from_bounds(
                strip_left, strip_bottom,
                strip_right, strip_top,
                full_w, actual_h
            )

            dst_strip = np.zeros(
                (band_count, actual_h, full_w), dtype=np.uint8
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

            strip_img = np.transpose(dst_strip, (1, 2, 0))

            for local_row in range(
                0, actual_h - tile_size + 1, stride
            ):
                for col in range(0, full_w - tile_size + 1, stride):
                    tile = strip_img[
                        local_row:local_row + tile_size,
                        col:col + tile_size
                    ]
                    if np.mean(tile) < 5:
                        continue

                    inp = preprocess_tile(tile).to(device)
                    with torch.no_grad():
                        logits = model(inp)
                        probs  = torch.softmax(logits, dim=1)
                        probs  = probs.squeeze(0).cpu().numpy()

                    gr = row_start + local_row
                    prob_sum[:, gr:gr+tile_size, col:col+tile_size] \
                        += probs * gauss
                    weight_sum[gr:gr+tile_size, col:col+tile_size] \
                        += gauss

    weight_sum = np.maximum(weight_sum, 1e-8)
    for c in range(4):
        prob_sum[c] /= weight_sum

    return np.argmax(prob_sum, axis=0).astype(np.uint8)


# ── VECTORIZATION ─────────────────────────────────────────────────────────────
def raster_to_polygons(
    mask: np.ndarray,
    class_value: int,
    transform,
    crs,
    min_area_m2: float,
) -> gpd.GeoDataFrame:
    """Converts raster mask class to vector polygons."""
    binary = (mask == class_value).astype(np.uint8)
    if binary.sum() == 0:
        return gpd.GeoDataFrame()

    geoms  = []
    for geom_dict, val in shapes(binary, transform=transform):
        if val == 1:
            geom = shape(geom_dict)
            geom = geom.simplify(
                CFG["simplify_tolerance"],
                preserve_topology=True
            )
            if geom.area >= min_area_m2:
                geoms.append(geom)

    if not geoms:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(
        {"geometry": geoms},
        crs=crs
    )
    return gdf


def classify_roofs(
    buildings_gdf: gpd.GeoDataFrame,
    ori_path: str,
    efficientnet_model,
    device,
) -> gpd.GeoDataFrame:
    """Classifies each building's roof type using EfficientNetV2."""
    if len(buildings_gdf) == 0:
        return buildings_gdf

    roof_types   = []
    confidences  = []

    with rasterio.open(ori_path) as src:
        src_crs = CRS.from_wkt(src.crs.to_wkt())
        bldgs   = buildings_gdf.to_crs(src_crs)

        for _, row in tqdm(
            bldgs.iterrows(),
            total=len(bldgs),
            desc="  Classifying roofs",
            leave=False
        ):
            geom    = row.geometry
            bounds  = geom.bounds
            px_size = abs(src.transform.a)
            pad     = CFG["crop_padding_px"] * px_size

            from rasterio.mask import mask as rio_mask
            from shapely.geometry import box

            padded = box(
                bounds[0]-pad, bounds[1]-pad,
                bounds[2]+pad, bounds[3]+pad
            )
            try:
                crop, _ = rio_mask(
                    src, [mapping(padded)],
                    crop=True, nodata=0, filled=True
                )
                crop = np.transpose(crop[:3], (1, 2, 0))
                if crop.shape[0] < 4 or crop.shape[1] < 4:
                    raise ValueError("too small")
                if np.mean(crop) < 5:
                    raise ValueError("mostly black")

                aug   = ROOF_TRANSFORM(image=crop.astype(np.uint8))
                inp   = aug["image"].unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = efficientnet_model(inp)
                    probs  = torch.softmax(logits, dim=1)
                    pred   = probs.argmax(dim=1).item()
                    conf   = probs.max().item()

                roof_types.append(CFG["roof_classes"][pred])
                confidences.append(round(conf, 3))

            except Exception:
                roof_types.append("Others")
                confidences.append(0.0)

    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["roof_type"]  = roof_types
    buildings_gdf["confidence"] = confidences
    return buildings_gdf


# ── GEOPACKAGE OUTPUT ─────────────────────────────────────────────────────────
def save_geopackage(
    buildings_gdf: gpd.GeoDataFrame,
    roads_gdf:     gpd.GeoDataFrame,
    water_gdf:     gpd.GeoDataFrame,
    utility_gdf:   gpd.GeoDataFrame,
    output_path:   str,
    village_name:  str,
):
    """Saves all feature layers to a single OGC GeoPackage."""
    target_crs = "EPSG:4326"
    today      = datetime.now().strftime("%Y-%m-%d")

    def prep(gdf, feature_type):
        if len(gdf) == 0:
            return gdf
        gdf = gdf.to_crs(target_crs)
        gdf["feature_type"]     = feature_type
        gdf["village_name"]     = village_name
        gdf["extraction_date"]  = today
        gdf["area_m2"]          = gdf.to_crs(
            "EPSG:32644"
        ).geometry.area.round(2)
        return gdf

    buildings_gdf = prep(buildings_gdf, "building")
    roads_gdf     = prep(roads_gdf,     "road")
    water_gdf     = prep(water_gdf,     "water")
    utility_gdf   = prep(utility_gdf,   "utility")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write each layer
    buildings_gdf.to_file(output_path, layer="buildings", driver="GPKG")
    roads_gdf.to_file(    output_path, layer="roads",     driver="GPKG")
    water_gdf.to_file(    output_path, layer="water",     driver="GPKG")
    utility_gdf.to_file(  output_path, layer="utilities", driver="GPKG")

    log.info(f"GeoPackage saved: {output_path}")
    log.info(f"  buildings: {len(buildings_gdf)}")
    log.info(f"  roads:     {len(roads_gdf)}")
    log.info(f"  water:     {len(water_gdf)}")
    log.info(f"  utilities: {len(utility_gdf)}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Load all models once
    segformer,    seg_cfg  = load_segformer(
        CFG["segformer_path"], device
    )
    deeplabv3,    dlab_cfg = load_deeplabv3(
        CFG["deeplabv3_path"], device
    )
    efficientnet, eff_cfg  = load_efficientnet(
        CFG["efficientnet_path"], device
    )

    for village in TEST_VILLAGES:
        name        = village["name"]
        ori_path    = village["ori_path"]
        working_crs = village["working_crs"]

        log.info(f"\n{'='*60}")
        log.info(f"Processing: {name}")
        log.info(f"{'='*60}")

        if not Path(ori_path).exists():
            log.error(f"ORI not found: {ori_path} — skipping")
            continue

        # Step 1 — SegFormer: building mask
        log.info("Step 1/4: SegFormer — building segmentation...")
        bldg_mask, transform, crs = predict_full_image_segformer(
            segformer, ori_path, working_crs, device
        )

        # Step 2 — DeepLabV3+: roads/water/utilities mask
        log.info("Step 2/4: DeepLabV3+ — infrastructure...")
        infra_mask = predict_full_image_deeplabv3(
            deeplabv3, ori_path, working_crs, device
        )

        # Step 3 — Vectorize all masks
        log.info("Step 3/4: Vectorizing predictions...")
        buildings_gdf = raster_to_polygons(
            bldg_mask, 1, transform, crs,
            CFG["min_building_area_m2"]
        )
        roads_gdf     = raster_to_polygons(
            infra_mask, 1, transform, crs,
            CFG["min_road_area_m2"]
        )
        water_gdf     = raster_to_polygons(
            infra_mask, 2, transform, crs,
            CFG["min_water_area_m2"]
        )
        utility_gdf   = raster_to_polygons(
            infra_mask, 3, transform, crs,
            CFG["min_utility_area_m2"]
        )

        log.info(
            f"  Raw polygons — "
            f"buildings={len(buildings_gdf)} "
            f"roads={len(roads_gdf)} "
            f"water={len(water_gdf)} "
            f"utilities={len(utility_gdf)}"
        )

        # Step 4 — Classify roof types
        log.info("Step 4/4: EfficientNetV2 — roof classification...")
        if len(buildings_gdf) > 0:
            buildings_gdf = classify_roofs(
                buildings_gdf, ori_path, efficientnet, device
            )

        # Save GeoPackage
        out_path = (
            Path(CFG["output_dir"]) / f"{name}_prediction.gpkg"
        )
        save_geopackage(
            buildings_gdf, roads_gdf, water_gdf, utility_gdf,
            str(out_path), name
        )

    log.info("\nAll villages processed.")
    log.info(f"GeoPackages saved to: {CFG['output_dir']}")


if __name__ == "__main__":
    main()