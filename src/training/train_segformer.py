"""
SegFormer-b2 training — SVAMITVA feature extraction
Classes: 0=background 1=building 2=road 3=water 4=utility

Fixes applied:
  - Removed double division bug (img/255 + A.Normalize)
  - Memory-safe incremental IoU metric
  - img kept as uint8 for albumentations

Run: python src/training/train_segformer.py
"""

import logging
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import rasterio
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import mlflow

# ── CONFIG ────────────────────────────────────────────────────────────────────
CFG = {
    "model_name":    "nvidia/mit-b2",
    "num_classes":   5,
    "image_size":    512,
    "batch_size":    16,
    "epochs":        80,
    "lr":            6e-5,
    "weight_decay":  0.01,
    "warmup_epochs": 5,
    "early_stop":    15,

    "train_dir":  "/home/kalki/data/tiles/train",
    "val_dir":    "/home/kalki/data/tiles/val",
    "output_dir": "/home/kalki/models/segformer",

    "class_weights":  [0.1, 2.0, 3.0, 4.0, 6.0],
    "class_names":    ["background","building","road","water","utility"],

    "mlflow_experiment": "segformer-b2-svamitva",
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
    def __init__(self, root_dir: str, transform=None):
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
            img = src.read()                     # (C, H, W)
            img = np.transpose(img, (1, 2, 0))   # (H, W, C)
            # Keep as uint8 — Albumentations handles normalization
            img = img.astype(np.uint8)

        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.int64)  # (H, W)

        # Clamp mask to valid class range
        mask = np.clip(mask, 0, CFG["num_classes"] - 1)

        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img  = aug["image"]   # tensor (C, H, W) float
            mask = aug["mask"]    # tensor (H, W) long
        else:
            img  = torch.from_numpy(
                img.transpose(2, 0, 1)
            ).float() / 255.0
            mask = torch.from_numpy(mask).long()

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
            # A.Normalize divides by 255 internally
            # Do NOT divide by 255 manually before this
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


# ── METRICS — memory safe ─────────────────────────────────────────────────────
class IoUMetric:
    """
    Incremental IoU — updates per batch, never stores full arrays.
    Fixes the RAM accumulation bug.
    """
    def __init__(self, num_classes: int = 5):
        self.num_classes  = num_classes
        self.intersection = np.zeros(num_classes, dtype=np.float64)
        self.union        = np.zeros(num_classes, dtype=np.float64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        p = preds.cpu().numpy().flatten()
        t = targets.cpu().numpy().flatten()
        for c in range(self.num_classes):
            pc = (p == c)
            tc = (t == c)
            self.intersection[c] += np.logical_and(pc, tc).sum()
            self.union[c]        += np.logical_or(pc, tc).sum()

    def get_miou(self) -> float:
        ious = []
        for c in range(self.num_classes):
            if self.union[c] > 0:
                ious.append(self.intersection[c] / self.union[c])
        return float(np.mean(ious)) if ious else 0.0

    def get_per_class_iou(self) -> dict:
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
    def __init__(self, class_weights, num_classes=5):
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
def build_model(num_classes: int):
    log.info("Loading SegFormer-b2 pretrained weights...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        CFG["model_name"],
        use_safetensors=True
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        id2label={str(i): n for i, n in enumerate(CFG["class_names"])},
        label2id={n: str(i) for i, n in enumerate(CFG["class_names"])},
    )
    return model


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    metric     = IoUMetric(CFG["num_classes"])

    for imgs, masks in tqdm(loader, desc="  Train", leave=False):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        outputs = model(pixel_values=imgs)
        logits  = F.interpolate(
            outputs.logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = criterion(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # Incremental metric update — no array accumulation
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            metric.update(preds, masks)

    return total_loss / len(loader), metric.get_miou()


# ── VALIDATE ──────────────────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metric     = IoUMetric(CFG["num_classes"])

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="  Val  ", leave=False):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            outputs = model(pixel_values=imgs)
            logits  = F.interpolate(
                outputs.logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = criterion(logits, masks)
            total_loss += loss.item()

            # Incremental update — no RAM accumulation
            preds = logits.argmax(dim=1)
            metric.update(preds, masks)

    return total_loss / len(loader), metric.get_miou(), \
           metric.get_per_class_iou()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | GPU: {torch.cuda.get_device_name(0)}")
    log.info(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB"
    )

    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = TileDataset(
        CFG["train_dir"], get_transforms(train=True)
    )
    val_ds   = TileDataset(
        CFG["val_dir"],   get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    log.info(
        f"Train: {len(train_ds)} tiles | "
        f"Val: {len(val_ds)} tiles"
    )

    # Model, loss, optimizer
    model     = build_model(CFG["num_classes"]).to(device)
    criterion = DiceCELoss(CFG["class_weights"], CFG["num_classes"])
    optimizer = AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )

    # Warmup then cosine decay
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=CFG["warmup_epochs"],
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=CFG["epochs"] - CFG["warmup_epochs"],
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[CFG["warmup_epochs"]],
    )

    best_miou  = 0.0
    no_improve = 0
    best_path  = Path(CFG["output_dir"]) / "best_segformer.pth"

    mlflow.set_experiment(CFG["mlflow_experiment"])

    with mlflow.start_run(run_name="segformer-b2-fixed"):
        mlflow.log_params(CFG)

        for epoch in range(1, CFG["epochs"] + 1):
            t0 = time.time()

            tr_loss, tr_miou = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            va_loss, va_miou, va_cls = validate(
                model, val_loader, criterion, device
            )
            scheduler.step()
            elapsed = time.time() - t0

            log.info(
                f"Epoch {epoch:3d}/{CFG['epochs']} | "
                f"Train loss={tr_loss:.4f} mIoU={tr_miou:.4f} | "
                f"Val loss={va_loss:.4f} mIoU={va_miou:.4f} | "
                f"LR={scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.0f}s"
            )

            # Per-class IoU
            for cls_name, iou in va_cls.items():
                if iou is not None:
                    log.info(f"  {cls_name:12s}: IoU={iou:.4f}")

            mlflow.log_metrics({
                "train_loss": tr_loss,
                "train_miou": tr_miou,
                "val_loss":   va_loss,
                "val_miou":   va_miou,
                "lr":         scheduler.get_last_lr()[0],
            }, step=epoch)

            for cls_name, iou in va_cls.items():
                if iou is not None:
                    mlflow.log_metric(
                        f"val_iou_{cls_name}", iou, step=epoch
                    )

            if va_miou > best_miou:
                best_miou  = va_miou
                no_improve = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
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

        log.info(f"\nBest val mIoU: {best_miou:.4f}")
        log.info(f"Saved: {best_path}")
        mlflow.log_artifact(str(best_path))


if __name__ == "__main__":
    main()

