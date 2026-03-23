"""
SegFormer-b2 training for SVAMITVA feature extraction.
Classes: 0=background 1=building 2=road 3=water 4=utility

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
from torch.optim.lr_scheduler import CosineAnnealingLR
import rasterio
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig,
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# ── CONFIG ────────────────────────────────────────────────────────────────────
CFG = {
    "model_name":   "nvidia/mit-b2",
    "num_classes":  5,
    "image_size":   512,
    "batch_size":   16,
    "epochs":       80,
    "lr":           6e-5,
    "weight_decay": 0.01,
    "warmup_epochs":5,
    "early_stop":   15,

    "train_dir":    "/home/kalki/data/tiles/train",
    "val_dir":      "/home/kalki/data/tiles/val",
    "output_dir":   "/home/kalki/models/segformer",

    # Class weights — inverse frequency
    # bg=0.1 building=2.0 road=3.0 water=4.0 utility=6.0
    "class_weights": [0.1, 2.0, 3.0, 4.0, 6.0],

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
            img = src.read()                    # (C, H, W)
            img = np.transpose(img, (1, 2, 0))  # (H, W, C)
            img = img.astype(np.float32)

        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.int64)  # (H, W)

        # Normalize to [0, 1]
        img = img / 255.0

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img  = aug["image"]
            mask = aug["mask"]
        else:
            img  = torch.from_numpy(img.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).long()

        return img, mask


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
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ])


# ── LOSS ──────────────────────────────────────────────────────────────────────
class DiceCELoss(nn.Module):
    def __init__(self, class_weights, num_classes=5):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32).cuda()
        self.ce   = nn.CrossEntropyLoss(weight=w)
        self.nc   = num_classes

    def dice_loss(self, pred, target, smooth=1.0):
        pred   = torch.softmax(pred, dim=1)
        total  = 0.0
        for c in range(self.nc):
            p  = pred[:, c]
            t  = (target == c).float()
            inter = (p * t).sum()
            total += 1 - (2*inter + smooth) / (p.sum() + t.sum() + smooth)
        return total / self.nc

    def forward(self, pred, target):
        return 0.5 * self.ce(pred, target) + \
               0.5 * self.dice_loss(pred, target)


# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_miou(preds, targets, num_classes=5):
    ious = []
    preds   = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    for c in range(num_classes):
        pred_c = preds == c
        tgt_c  = targets == c
        inter  = np.logical_and(pred_c, tgt_c).sum()
        union  = np.logical_or(pred_c,  tgt_c).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# ── MODEL ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int):
    log.info(f"Loading SegFormer-b2 pretrained weights...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        CFG["model_name"],
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        id2label={str(i): n for i, n in enumerate(
            ["background","building","road","water","utility"])},
        label2id={n: str(i) for i, n in enumerate(
            ["background","building","road","water","utility"])},
    )
    return model


# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_targets = []

    for imgs, masks in tqdm(loader, desc="  Train", leave=False):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        outputs = model(pixel_values=imgs)
        logits  = outputs.logits  # (B, C, H/4, W/4)

        # Upsample to original size
        logits = F.interpolate(
            logits,
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

        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach())
        all_targets.append(masks.detach())

    avg_loss = total_loss / len(loader)
    miou     = compute_miou(
        torch.cat(all_preds),
        torch.cat(all_targets),
        CFG["num_classes"]
    )
    return avg_loss, miou


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_targets = []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="  Val  ", leave=False):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            outputs = model(pixel_values=imgs)
            logits  = outputs.logits
            logits  = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = criterion(logits, masks)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.append(preds)
            all_targets.append(masks)

    avg_loss = total_loss / len(loader)
    miou     = compute_miou(
        torch.cat(all_preds),
        torch.cat(all_targets),
        CFG["num_classes"]
    )
    return avg_loss, miou


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = TileDataset(CFG["train_dir"], get_transforms(train=True))
    val_ds   = TileDataset(CFG["val_dir"],   get_transforms(train=False))

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

    # Model
    model = build_model(CFG["num_classes"]).to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Loss, optimizer, scheduler
    criterion = DiceCELoss(CFG["class_weights"], CFG["num_classes"])
    optimizer = AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CFG["epochs"],
        eta_min=1e-6,
    )

    # MLflow
    mlflow.set_experiment(CFG["mlflow_experiment"])

    best_miou     = 0.0
    no_improve    = 0
    best_path     = Path(CFG["output_dir"]) / "best_segformer.pth"

    with mlflow.start_run(run_name="segformer-b2-run1"):
        mlflow.log_params(CFG)

        for epoch in range(1, CFG["epochs"] + 1):
            t0 = time.time()

            train_loss, train_miou = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_miou = validate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            elapsed = time.time() - t0

            log.info(
                f"Epoch {epoch:3d}/{CFG['epochs']} | "
                f"Train loss={train_loss:.4f} mIoU={train_miou:.4f} | "
                f"Val loss={val_loss:.4f} mIoU={val_miou:.4f} | "
                f"LR={scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.0f}s"
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_miou": train_miou,
                "val_loss":   val_loss,
                "val_miou":   val_miou,
                "lr":         scheduler.get_last_lr()[0],
            }, step=epoch)

            # Save best model
            if val_miou > best_miou:
                best_miou  = val_miou
                no_improve = 0
                torch.save({
                    "epoch":      epoch,
                    "model_state": model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "val_miou":    val_miou,
                    "val_loss":    val_loss,
                    "cfg":         CFG,
                }, best_path)
                log.info(f"  ✓ New best mIoU: {best_miou:.4f} → saved")
                mlflow.log_metric("best_val_miou", best_miou, step=epoch)
            else:
                no_improve += 1
                log.info(
                    f"  No improvement ({no_improve}/{CFG['early_stop']})"
                )

            # Early stopping
            if no_improve >= CFG["early_stop"]:
                log.info(f"Early stopping at epoch {epoch}")
                break

        log.info(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")
        log.info(f"Model saved: {best_path}")
        mlflow.log_artifact(str(best_path))


if __name__ == "__main__":
    main()