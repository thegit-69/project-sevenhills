"""
DeepLabV3+ EfficientNet-B4 training — Roads, Water, Utilities
Classes: 0=background  1=road  2=water  3=utility

Uses same tiles as SegFormer but remaps mask:
  original 0 (bg)       → 0 (background)
  original 1 (building) → 0 (background)  ← buildings ignored
  original 2 (road)     → 1 (road)
  original 3 (water)    → 2 (water)
  original 4 (utility)  → 3 (utility)

Run: python src/training/train_deeplabv3.py
"""

import logging
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import rasterio
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import mlflow

# ── CONFIG ────────────────────────────────────────────────────────────────────
CFG = {
    "encoder":       "efficientnet-b4",
    "encoder_weights": "imagenet",
    "num_classes":   4,           # bg, road, water, utility
    "image_size":    512,
    "batch_size":    16,
    "epochs":        60,
    "lr":            1e-4,
    "weight_decay":  1e-4,
    "warmup_epochs": 5,
    "early_stop":    15,

    "train_dir":  "/home/kalki/data/tiles/train",
    "val_dir":    "/home/kalki/data/tiles/val",
    "output_dir": "/home/kalki/models/deeplabv3",

    # Class weights — utility is very sparse, boost it heavily
    # background=0.1, road=3.0, water=4.0, utility=8.0
    "class_weights": [0.1, 3.0, 4.0, 8.0],

    "class_names": ["background", "road", "water", "utility"],

    # Mask remapping from original 5-class to this 4-class
    # original_value → new_value
    "mask_remap": {
        0: 0,   # background → background
        1: 0,   # building   → background (not our job)
        2: 1,   # road       → road
        3: 2,   # water      → water
        4: 3,   # utility    → utility
    },

    "mlflow_experiment": "deeplabv3-roads-water-utility",
    "seed": 42,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ── DATASET ───────────────────────────────────────────────────────────────────
class TileDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_paths  = []
        self.mask_paths = []
        self.transform  = transform

        root = Path(root_dir)
        for village_dir in sorted(root.iterdir()):
            img_dir  = village_dir / "img"
            mask_dir = village_dir / "mask"
            if not img_dir.exists():
                continue
            for img_f in sorted(img_dir.glob("*.tif")):
                mask_f = mask_dir / img_f.name.replace("img_", "mask_")
                if mask_f.exists():
                    self.img_paths.append(img_f)
                    self.mask_paths.append(mask_f)

        log.info(f"Dataset {root_dir}: {len(self.img_paths)} tiles")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.img_paths[idx]) as src:
            img = src.read()
            img = np.transpose(img, (1, 2, 0))

        # Force 3 channels
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] > 3:
            img = img[:, :, :3]

        img = img.astype(np.uint8)

        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1)

        # Remap mask from 5-class to 4-class using an ultra-fast NumPy lookup table
        # Original: 0=bg, 1=building, 2=road, 3=water, 4=utility
        # New Map:  0=bg, 0=bg,       1=road, 2=water, 3=utility
        remap_table = np.array([0, 0, 1, 2, 3], dtype=np.int64)
        new_mask = remap_table[mask]
        
        if self.transform:
            aug  = self.transform(image=img, mask=new_mask)
            img  = aug["image"]
            mask = aug["mask"]
        else:
            img  = torch.from_numpy(
                img.transpose(2, 0, 1)
            ).float() / 255.0
            mask = torch.from_numpy(new_mask).long()

        return img, mask.long()


def get_transforms(train=True):
    if train:
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2, p=0.4
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10, p=0.3
            ),
            A.GaussNoise(p=0.2),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])


# ── METRICS ───────────────────────────────────────────────────────────────────
class IoUMetric:
    def __init__(self, num_classes):
        self.num_classes  = num_classes
        self.intersection = np.zeros(num_classes, dtype=np.float64)
        self.union        = np.zeros(num_classes, dtype=np.float64)

    def update(self, preds, targets):
        p = preds.cpu().numpy().flatten()
        t = targets.cpu().numpy().flatten()
        for c in range(self.num_classes):
            pc = (p == c)
            tc = (t == c)
            self.intersection[c] += np.logical_and(pc, tc).sum()
            self.union[c]        += np.logical_or(pc, tc).sum()

    def get_miou(self):
        # Exclude background from mIoU — only care about road/water/utility
        ious = []
        for c in range(1, self.num_classes):   # start from 1, skip background
            if self.union[c] > 0:
                ious.append(self.intersection[c] / self.union[c])
        return float(np.mean(ious)) if ious else 0.0

    def get_per_class(self):
        result = {}
        for c in range(self.num_classes):
            if self.union[c] > 0:
                result[CFG["class_names"][c]] = round(
                    float(self.intersection[c] / self.union[c]), 4
                )
            else:
                result[CFG["class_names"][c]] = None
        return result


# ── LOSS ──────────────────────────────────────────────────────────────────────
class DiceCELoss(nn.Module):
    def __init__(self, class_weights, num_classes):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32).cuda()
        self.ce = nn.CrossEntropyLoss(weight=w)
        self.nc = num_classes

    def dice_loss(self, pred, target, smooth=1.0):
        pred  = torch.softmax(pred, dim=1)
        total = 0.0
        for c in range(self.nc):
            p     = pred[:, c]
            t     = (target == c).float()
            inter = (p * t).sum()
            total += 1 - (2*inter + smooth) / (p.sum() + t.sum() + smooth)
        return total / self.nc

    def forward(self, pred, target):
        return 0.5 * self.ce(pred, target) + \
               0.5 * self.dice_loss(pred, target)


# ── MODEL ─────────────────────────────────────────────────────────────────────
def build_model(num_classes):
    log.info(
        f"Building DeepLabV3+ with {CFG['encoder']} "
        f"backbone (pretrained={CFG['encoder_weights']})..."
    )
    model = smp.DeepLabV3Plus(
        encoder_name=CFG["encoder"],
        encoder_weights=CFG["encoder_weights"],
        in_channels=3,
        classes=num_classes,
    )
    params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model parameters: {params:.1f}M")
    return model


# ── TRAIN / VALIDATE ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    metric     = IoUMetric(CFG["num_classes"])

    for imgs, masks in tqdm(loader, desc="  Train", leave=False):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(imgs)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        with torch.no_grad():
            metric.update(logits.argmax(dim=1), masks)

    return total_loss / len(loader), metric.get_miou()


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metric     = IoUMetric(CFG["num_classes"])

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="  Val  ", leave=False):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss   = criterion(logits, masks)
            total_loss += loss.item()
            metric.update(logits.argmax(dim=1), masks)

    return total_loss / len(loader), metric.get_miou(), \
           metric.get_per_class()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        f"GPU: {torch.cuda.get_device_name(0)} | "
        f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB"
    )

    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Check class distribution before training
    log.info("Checking mask class distribution on sample tiles...")
    sample_dir = Path(CFG["train_dir"])
    sample_files = []
    for vd in sample_dir.iterdir():
        mdir = vd / "mask"
        if mdir.exists():
            sample_files.extend(list(mdir.glob("*.tif"))[:20])
        if len(sample_files) >= 100:
            break

    class_px = {i: 0 for i in range(5)}
    for f in sample_files[:100]:
        with rasterio.open(f) as s:
            d = s.read(1)
            for c in range(5):
                class_px[c] += int(np.sum(d == c))
    total = sum(class_px.values()) or 1
    orig_names = {0:"bg", 1:"building", 2:"road", 3:"water", 4:"utility"}
    log.info("Original mask distribution (sample 100 tiles):")
    for c, px in class_px.items():
        log.info(f"  class {c} ({orig_names[c]}): {px/total*100:.1f}%")

    # Datasets
    train_ds = TileDataset(CFG["train_dir"], get_transforms(True))
    val_ds   = TileDataset(CFG["val_dir"],   get_transforms(False))

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"],
        shuffle=True, num_workers=6,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"],
        shuffle=False, num_workers=6,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=2,
    )

    model     = build_model(CFG["num_classes"]).to(device)
    criterion = DiceCELoss(CFG["class_weights"], CFG["num_classes"])
    optimizer = Adam(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda")

    warmup = LinearLR(
        optimizer, start_factor=0.1,
        end_factor=1.0, total_iters=CFG["warmup_epochs"]
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=CFG["epochs"] - CFG["warmup_epochs"],
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[CFG["warmup_epochs"]]
    )

    best_miou  = 0.0
    no_improve = 0
    best_path  = Path(CFG["output_dir"]) / "best_deeplabv3.pth"

    mlflow.set_experiment(CFG["mlflow_experiment"])

    with mlflow.start_run(run_name="deeplabv3-efficientnet-b4"):
        mlflow.log_params(CFG)

        for epoch in range(1, CFG["epochs"] + 1):
            t0 = time.time()

            tr_loss, tr_miou         = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            va_loss, va_miou, va_cls = validate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            log.info(
                f"Epoch {epoch:3d}/{CFG['epochs']} | "
                f"Train loss={tr_loss:.4f} mIoU={tr_miou:.4f} | "
                f"Val loss={va_loss:.4f} mIoU={va_miou:.4f} | "
                f"LR={scheduler.get_last_lr()[0]:.2e} | "
                f"{time.time()-t0:.0f}s"
            )
            for cls, iou in va_cls.items():
                if iou is not None:
                    log.info(f"  {cls:12s}: IoU={iou:.4f}")

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_miou": tr_miou,
                "val_loss":   va_loss, "val_miou":   va_miou,
                "lr":         scheduler.get_last_lr()[0],
            }, step=epoch)
            for cls, iou in va_cls.items():
                if iou is not None:
                    mlflow.log_metric(
                        f"val_iou_{cls}", iou, step=epoch
                    )

            if va_miou > best_miou:
                best_miou  = va_miou
                no_improve = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_miou":    va_miou,
                    "val_cls":     va_cls,
                    "cfg":         CFG,
                }, best_path)
                log.info(f"  ✓ Best mIoU: {best_miou:.4f} → saved")
                mlflow.log_metric("best_val_miou", best_miou, step=epoch)
            else:
                no_improve += 1
                log.info(
                    f"  No improvement "
                    f"({no_improve}/{CFG['early_stop']})"
                )

            if no_improve >= CFG["early_stop"]:
                log.info(f"Early stopping at epoch {epoch}")
                break

        log.info(
            f"Training complete. "
            f"Best mIoU (road+water+utility): {best_miou:.4f}"
        )
        log.info(f"Model saved: {best_path}")
        mlflow.log_artifact(str(best_path))


if __name__ == "__main__":
    main()
