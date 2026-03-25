"""
Inference pipeline — SVAMITVA Feature Extraction (Enterprise Optimized)
Runs SegFormer (buildings) + DeepLabV3+ (roads/water/utilities)
Optimizations: np.memmap for zero-RAM overhead, GPU Batching, Sequential Cleanup
"""

import logging
import warnings
warnings.filterwarnings("ignore")
import os
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
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
import gc
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CFG = {
    "segformer_path":    "/home/kalki/models/segformer_binary/best_segformer.pth",
    "deeplabv3_path":    "/home/kalki/models/deeplabv3/best_deeplabv3.pth",
    "efficientnet_path": "/home/kalki/models/efficientnet/best_efficientnet.pth",
    "output_dir":        "/home/kalki/data/predictions",

    "tile_size":    512,
    "stride":       256,
    "strip_height": 3000,
    "batch_size":   16,  # GPU Batching enabled!

    "simplify_tolerance": 0.5,
    "min_building_area_m2": 8.0,
    "min_road_area_m2":     5.0,
    "min_water_area_m2":    10.0,
    "min_utility_area_m2":  2.0,
    "crop_padding_px": 10,

    "roof_classes": {0: "RCC", 1: "Tiled", 2: "Tin", 3: "Others"},
}

# ── TEST VILLAGES ─────────────────────────────────────────────────────────────
TEST_VILLAGES = [
    {
        "name":        "DIWANA_BARNALA",
        "ori_path":    "/home/kalki/data/raw/test/PB/live_demo_2/DIWANA_BARNALA_40082_ORTHO/DIWANA_BARNALA_40082_ORTHO.tif",
        "working_crs": "EPSG:32644",
    },
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
VAL_TRANSFORM = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])

ROOF_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def load_segformer(path: str, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    cfg   = ckpt["cfg"]
    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg["model_name"], use_safetensors=True, num_labels=cfg["num_classes"],
        ignore_mismatched_sizes=True, id2label={str(i): n for i, n in enumerate(cfg["class_names"])},
        label2id={n: str(i) for i, n in enumerate(cfg["class_names"])},
    )
    model.load_state_dict(ckpt["model_state"])
    return model.eval().to(device), cfg

def load_deeplabv3(path: str, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    cfg   = ckpt["cfg"]
    model = smp.DeepLabV3Plus(encoder_name=cfg["encoder"], encoder_weights=None, in_channels=3, classes=cfg["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    return model.eval().to(device), cfg

def load_efficientnet(path: str, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    cfg   = ckpt["cfg"]
    model = timm.create_model(cfg["model_name"], pretrained=False, num_classes=cfg["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    return model.eval().to(device), cfg

def preprocess_tile(tile: np.ndarray) -> torch.Tensor:
    if tile.shape[2] == 1: tile = np.repeat(tile, 3, axis=2)
    elif tile.shape[2] > 3: tile = tile[:, :, :3]
    return VAL_TRANSFORM(image=tile.astype(np.uint8))["image"]

# ── INFERENCE ─────────────────────────────────────────────────────────
def predict_full_image_segformer(model, ori_path: str, working_crs: str, device):
    with rasterio.open(ori_path) as src:
        transform, full_w, full_h = calculate_default_transform(src.crs, working_crs, src.width, src.height, *src.bounds)
        band_count = min(src.count, 3)

    tile_sz = CFG["tile_size"]
    stride  = CFG["stride"]
    
    # MEMMAP DISK-BACKED ARRAYS (Prevents OOM Crashes)
    prob_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
    wt_tmp   = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
    prob_sum = np.memmap(prob_tmp.name, dtype=np.float16, mode='w+', shape=(2, full_h, full_w))
    weight_sum = np.memmap(wt_tmp.name, dtype=np.float16, mode='w+', shape=(full_h, full_w))

    y_idx, x_idx = np.mgrid[:tile_sz, :tile_sz]
    gauss = np.exp(-((y_idx - tile_sz//2)**2 + (x_idx - tile_sz//2)**2) / (2 * (tile_sz * 0.25)**2)).astype(np.float16)

    strip_bounds = list(range(0, full_h, CFG["strip_height"]))
    if strip_bounds[-1] < full_h: strip_bounds.append(full_h)

    with rasterio.open(ori_path) as src:
        for s_idx in tqdm(range(len(strip_bounds) - 1), desc="  SegFormer strips", leave=False):
            actual_h = strip_bounds[s_idx + 1] - strip_bounds[s_idx]
            strip_tf = from_bounds(transform.c, transform.f + strip_bounds[s_idx+1]*transform.e, transform.c + full_w*transform.a, transform.f + strip_bounds[s_idx]*transform.e, full_w, actual_h)
            
            dst_strip = np.zeros((band_count, actual_h, full_w), dtype=np.uint8)
            for band in range(1, band_count + 1):
                reproject(source=rasterio.band(src, band), destination=dst_strip[band - 1], src_transform=src.transform, src_crs=src.crs, dst_transform=strip_tf, dst_crs=working_crs, resampling=Resampling.nearest)
            strip_img = np.transpose(dst_strip, (1, 2, 0))

            batch_tensors, batch_coords = [], []
            
            def process_batch():
                if not batch_tensors: return
                batch_inp = torch.stack(batch_tensors).to(device)
                with torch.no_grad():
                    logits = F.interpolate(model(pixel_values=batch_inp).logits, size=(tile_sz, tile_sz), mode="bilinear", align_corners=False)
                    probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float16)
                
                for i, (gr, gc_col) in enumerate(batch_coords):
                    prob_sum[:, gr:gr+tile_sz, gc_col:gc_col+tile_sz] += probs[i] * gauss
                    weight_sum[gr:gr+tile_sz, gc_col:gc_col+tile_sz] += gauss
                batch_tensors.clear(); batch_coords.clear()

            for local_row in range(0, actual_h - tile_sz + 1, stride):
                for col in range(0, full_w - tile_sz + 1, stride):
                    tile = strip_img[local_row:local_row+tile_sz, col:col+tile_sz]
                    if np.mean(tile) < 5: continue
                    batch_tensors.append(preprocess_tile(tile))
                    batch_coords.append((strip_bounds[s_idx] + local_row, col))
                    
                    if len(batch_tensors) == CFG["batch_size"]: process_batch()
            process_batch() # flush remainder

    np.maximum(weight_sum, 1e-8, out=weight_sum)
    for c in range(2): prob_sum[c] /= weight_sum
    pred = np.argmax(prob_sum, axis=0).astype(np.uint8)

    # SECURE MEMORY CLEANUP
    del prob_sum, weight_sum
    gc.collect()
    os.remove(prob_tmp.name); os.remove(wt_tmp.name)
    
    return pred, transform, CRS.from_epsg(int(working_crs.split(":")[1]))

def predict_full_image_deeplabv3(model, ori_path: str, working_crs: str, device):
    with rasterio.open(ori_path) as src:
        transform, full_w, full_h = calculate_default_transform(src.crs, working_crs, src.width, src.height, *src.bounds)
        band_count = min(src.count, 3)

    tile_sz = CFG["tile_size"]
    stride  = CFG["stride"]
    
    # MEMMAP DISK-BACKED ARRAYS
    prob_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
    wt_tmp   = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
    prob_sum = np.memmap(prob_tmp.name, dtype=np.float16, mode='w+', shape=(4, full_h, full_w))
    weight_sum = np.memmap(wt_tmp.name, dtype=np.float16, mode='w+', shape=(full_h, full_w))

    y_idx, x_idx = np.mgrid[:tile_sz, :tile_sz]
    gauss = np.exp(-((y_idx - tile_sz//2)**2 + (x_idx - tile_sz//2)**2) / (2 * (tile_sz * 0.25)**2)).astype(np.float16)

    strip_bounds = list(range(0, full_h, CFG["strip_height"]))
    if strip_bounds[-1] < full_h: strip_bounds.append(full_h)

    with rasterio.open(ori_path) as src:
        for s_idx in tqdm(range(len(strip_bounds) - 1), desc="  DeepLabV3+ strips", leave=False):
            actual_h = strip_bounds[s_idx + 1] - strip_bounds[s_idx]
            strip_tf = from_bounds(transform.c, transform.f + strip_bounds[s_idx+1]*transform.e, transform.c + full_w*transform.a, transform.f + strip_bounds[s_idx]*transform.e, full_w, actual_h)
            
            dst_strip = np.zeros((band_count, actual_h, full_w), dtype=np.uint8)
            for band in range(1, band_count + 1):
                reproject(source=rasterio.band(src, band), destination=dst_strip[band - 1], src_transform=src.transform, src_crs=src.crs, dst_transform=strip_tf, dst_crs=working_crs, resampling=Resampling.nearest)
            strip_img = np.transpose(dst_strip, (1, 2, 0))

            batch_tensors, batch_coords = [], []
            
            def process_batch():
                if not batch_tensors: return
                batch_inp = torch.stack(batch_tensors).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(batch_inp), dim=1).cpu().numpy().astype(np.float16)
                for i, (gr, gc_col) in enumerate(batch_coords):
                    prob_sum[:, gr:gr+tile_sz, gc_col:gc_col+tile_sz] += probs[i] * gauss
                    weight_sum[gr:gr+tile_sz, gc_col:gc_col+tile_sz] += gauss
                batch_tensors.clear(); batch_coords.clear()

            for local_row in range(0, actual_h - tile_sz + 1, stride):
                for col in range(0, full_w - tile_sz + 1, stride):
                    tile = strip_img[local_row:local_row+tile_sz, col:col+tile_sz]
                    if np.mean(tile) < 5: continue
                    batch_tensors.append(preprocess_tile(tile))
                    batch_coords.append((strip_bounds[s_idx] + local_row, col))
                    
                    if len(batch_tensors) == CFG["batch_size"]: process_batch()
            process_batch()

    np.maximum(weight_sum, 1e-8, out=weight_sum)
    for c in range(4): prob_sum[c] /= weight_sum
    pred = np.argmax(prob_sum, axis=0).astype(np.uint8)

    del prob_sum, weight_sum
    gc.collect()
    os.remove(prob_tmp.name); os.remove(wt_tmp.name)
    return pred


# ── VECTORIZATION & ROOFS ─────────────────────────────────────────────────────
def raster_to_polygons(mask: np.ndarray, class_value: int, transform, crs, min_area_m2: float) -> gpd.GeoDataFrame:
    binary = (mask == class_value).astype(np.uint8)
    if binary.sum() == 0: return gpd.GeoDataFrame()

    geoms = [shape(geom_dict).simplify(CFG["simplify_tolerance"], preserve_topology=True) 
             for geom_dict, val in shapes(binary, transform=transform) if val == 1]
    
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=crs)
    return gdf[gdf.geometry.area >= min_area_m2] if not gdf.empty else gpd.GeoDataFrame()

def classify_roofs(buildings_gdf: gpd.GeoDataFrame, ori_path: str, efficientnet_model, device) -> gpd.GeoDataFrame:
    if len(buildings_gdf) == 0: return buildings_gdf
    roof_types, confidences = [], []

    with rasterio.open(ori_path) as src:
        src_crs = CRS.from_wkt(src.crs.to_wkt())
        bldgs   = buildings_gdf.to_crs(src_crs)

        for _, row in tqdm(bldgs.iterrows(), total=len(bldgs), desc="  Classifying roofs", leave=False):
            pad = CFG["crop_padding_px"] * abs(src.transform.a)
            from shapely.geometry import box
            padded = box(row.geometry.bounds[0]-pad, row.geometry.bounds[1]-pad, row.geometry.bounds[2]+pad, row.geometry.bounds[3]+pad)
            try:
                from rasterio.mask import mask as rio_mask
                crop, _ = rio_mask(src, [mapping(padded)], crop=True, nodata=0, filled=True)
                crop = np.transpose(crop[:3], (1, 2, 0))
                if crop.shape[0] < 4 or np.mean(crop) < 5: raise ValueError
                
                inp = ROOF_TRANSFORM(image=crop.astype(np.uint8))["image"].unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(efficientnet_model(inp), dim=1)
                roof_types.append(CFG["roof_classes"][probs.argmax().item()])
                confidences.append(round(probs.max().item(), 3))
            except Exception:
                roof_types.append("Others")
                confidences.append(0.0)

    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["roof_type"], buildings_gdf["confidence"] = roof_types, confidences
    return buildings_gdf

# ── GEOPACKAGE OUTPUT ─────────────────────────────────────────────────────────
def save_geopackage(b_gdf, r_gdf, w_gdf, u_gdf, output_path, village_name):
    def prep(gdf, f_type):
        if len(gdf) == 0: return gdf
        gdf = gdf.to_crs("EPSG:4326")
        gdf["feature_type"], gdf["village_name"], gdf["extraction_date"] = f_type, village_name, datetime.now().strftime("%Y-%m-%d")
        gdf["area_m2"] = gdf.to_crs("EPSG:32644").geometry.area.round(2)
        return gdf

    b_gdf, r_gdf, w_gdf, u_gdf = prep(b_gdf, "building"), prep(r_gdf, "road"), prep(w_gdf, "water"), prep(u_gdf, "utility")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    for gdf, layer in [(b_gdf, "buildings"), (r_gdf, "roads"), (w_gdf, "water"), (u_gdf, "utilities")]:
        if len(gdf) > 0: gdf.to_file(output_path, layer=layer, driver="GPKG")
    log.info(f"GeoPackage saved: {output_path} | Bldgs: {len(b_gdf)}, Roads: {len(r_gdf)}, Water: {len(w_gdf)}, Utils: {len(u_gdf)}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segformer, _ = load_segformer(CFG["segformer_path"], device)
    deeplabv3, _ = load_deeplabv3(CFG["deeplabv3_path"], device)
    efficientnet, _ = load_efficientnet(CFG["efficientnet_path"], device)

    for village in TEST_VILLAGES:
        log.info(f"\n{'='*60}\nProcessing: {village['name']}\n{'='*60}")
        if not Path(village["ori_path"]).exists(): continue

        # SEQUENTIAL PROCESSING: Prevents holding two massive arrays in memory
        # 1. SegFormer Mask -> Vectorize -> Delete
        log.info("Step 1/4: SegFormer — building segmentation...")
        bldg_mask, transform, crs = predict_full_image_segformer(segformer, village["ori_path"], village["working_crs"], device)
        buildings_gdf = raster_to_polygons(bldg_mask, 1, transform, crs, CFG["min_building_area_m2"])
        
        del bldg_mask; gc.collect()  # Burn the drafting paper immediately
        
        # 2. DeepLab Mask -> Vectorize -> Delete
        log.info("Step 2/4: DeepLabV3+ — infrastructure...")
        infra_mask = predict_full_image_deeplabv3(deeplabv3, village["ori_path"], village["working_crs"], device)
        log.info("Step 3/4: Vectorizing infrastructure...")
        roads_gdf   = raster_to_polygons(infra_mask, 1, transform, crs, CFG["min_road_area_m2"])
        water_gdf   = raster_to_polygons(infra_mask, 2, transform, crs, CFG["min_water_area_m2"])
        utility_gdf = raster_to_polygons(infra_mask, 3, transform, crs, CFG["min_utility_area_m2"])
        
        del infra_mask; gc.collect() # Burn the drafting paper immediately

        # 3. Classify Roofs
        log.info("Step 4/4: EfficientNetV2 — roof classification...")
        buildings_gdf = classify_roofs(buildings_gdf, village["ori_path"], efficientnet, device)

        # 4. Save
        save_geopackage(buildings_gdf, roads_gdf, water_gdf, utility_gdf, str(Path(CFG["output_dir"]) / f"{village['name']}_prediction.gpkg"), village['name'])

if __name__ == "__main__":
    main()