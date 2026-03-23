"""
EfficientNetV2-S training for roof type classification.
Classes: 0=RCC  1=Tiled  2=Tin  3=Others

Run: python src/training/train_efficientnet.py
"""

import logging
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import rasterio
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import mlflow

# ── CONFIG ────────────────────────────────────────────────────────────────────
CFG = {
    "model_name":    "tf_efficientnetv2_s",
    "num_classes":   4,
    "image_size":    224,
    "batch_size":    64,
    "epochs":        50,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "early_stop":    10,

    "train_dir":  "/home/kalki/data/roof_crops/train",
    "val_dir":    "/home/kalki/data/roof_crops/val",
    "output_dir": "/home/kalki/models/efficientnet",

    # Inverse frequency weights from combined dataset
    # RCC=71.2%  Tiled=1.3%  Tin=23.6%  Others=3.9%
    "class_weights": [1.0, 54.8, 3.02, 18.26],

    "class_names": ["RCC", "Tiled", "Tin", "Others"],

    # Folder names → class indices
    "folder_to_class": {
        "1_RCC":    0,
        "2_Tiled":  1,
        "3_Tin":    2,
        "4_Others": 3,
    },

    "mlflow_experiment": "efficientnetv2-roof-svamitva",
    "seed": 42,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ── DATASET ───────────────────────────────────────────────────────────────────
class RoofCropDataset(Dataset):
    """
    Loads individual building roof crops.
    Folder structure:
      split/1_RCC/*.tif
      split/2_Tiled/*.tif
      split/3_Tin/*.tif
      split/4_Others/*.tif
    """
    def __init__(self, root_dir: str, transform=None):
        self.samples   = []   # (path, class_idx)
        self.transform = transform
        root = Path(root_dir)

        for folder_name, class_idx in CFG["folder_to_class"].items():
            folder = root / folder_name
            if not folder.exists():
                log.warning(f"Missing folder: {folder}")
                continue
            files = list(folder.glob("*.tif"))
            for f in files:
                self.samples.append((f, class_idx))
            log.info(
                f"  {folder_name}: {len(files)} crops "
                f"(class {class_idx} = {CFG['class_names'][class_idx]})"
            )

        log.info(f"Total: {len(self.samples)} crops from {root_dir}")

        # Class distribution
        counts = [0] * CFG["num_classes"]
        for _, c in self.samples:
            counts[c] += 1
        for i, cnt in enumerate(counts):
            pct = cnt / len(self.samples) * 100 if self.samples else 0
            log.info(
                f"  Class {i} ({CFG['class_names'][i]}): "
                f"{cnt} ({pct:.1f}%)"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Read GeoTIFF crop
        try:
            with rasterio.open(path) as src:
                img = src.read()  # (C, H, W)
                # Handle 1 or 3 bands
                if img.shape[0] == 1:
                    img = np.repeat(img, 3, axis=0)
                elif img.shape[0] > 3:
                    img = img[:3]
                img = np.transpose(img, (1, 2, 0))  # (H, W, C)
                img = img.astype(np.float32)
        except Exception as e:
            # Fallback — return black image
            img = np.zeros((64, 64, 3), dtype=np.float32)

        # Normalize to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0

        if self.transform:
            aug   = self.transform(image=img)
            img   = aug["image"]
        else:
            img   = torch.from_numpy(
                img.transpose(2, 0, 1)
            ).float()

        return img, torch.tensor(label, dtype=torch.long)


def get_transforms(train=True, image_size=224):
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=15, p=0.4
            ),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32), p=0.2
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ])


# ── MODEL ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int):
    log.info("Loading EfficientNetV2-S...")
    try:
        model = timm.create_model(
            CFG["model_name"],
            pretrained=True,
            num_classes=num_classes,
        )
        log.info("Loaded with pretrained ImageNet weights")
    except RuntimeError:
        log.warning(
            "Pretrained weights unavailable — "
            "trying tf_efficientnetv2_s..."
        )
        try:
            model = timm.create_model(
                "tf_efficientnetv2_s",
                pretrained=True,
                num_classes=num_classes,
            )
            log.info("Loaded tf_efficientnetv2_s with pretrained weights")
        except RuntimeError:
            log.warning(
                "No pretrained weights found — "
                "using random init. "
                "Model will still train but needs more epochs."
            )
            model = timm.create_model(
                CFG["model_name"],
                pretrained=False,
                num_classes=num_classes,
            )
    return model

# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_metrics(preds_all, labels_all, num_classes):
    """Computes per-class accuracy, precision, recall, F1."""
    preds_all  = np.array(preds_all)
    labels_all = np.array(labels_all)

    overall_acc = np.mean(preds_all == labels_all)

    per_class = {}
    for c in range(num_classes):
        tp = np.sum((preds_all == c) & (labels_all == c))
        fp = np.sum((preds_all == c) & (labels_all != c))
        fn = np.sum((preds_all != c) & (labels_all == c))

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        per_class[CFG["class_names"][c]] = {
            "precision": round(float(precision), 4),
            "recall":    round(float(recall),    4),
            "f1":        round(float(f1),        4),
        }

    return overall_acc, per_class


# ── TRAIN / VALIDATE ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss     = total_loss / len(loader)
    acc, per_cls = compute_metrics(
        all_preds, all_labels, CFG["num_classes"]
    )
    return avg_loss, acc, per_cls


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss     = total_loss / len(loader)
    acc, per_cls = compute_metrics(
        all_preds, all_labels, CFG["num_classes"]
    )
    return avg_loss, acc, per_cls


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Check val directory exists — if empty use train for val
    val_path = Path(CFG["val_dir"])
    has_val  = val_path.exists() and any(
        list((val_path / d).glob("*.tif"))
        for d in CFG["folder_to_class"]
        if (val_path / d).exists()
    )

    if not has_val:
        log.warning(
            "Val directory empty — using train dir for validation. "
            "Consider splitting train data."
        )
        CFG["val_dir"] = CFG["train_dir"]

    # Datasets
    log.info("Loading train dataset...")
    train_ds = RoofCropDataset(
        CFG["train_dir"],
        get_transforms(train=True, image_size=CFG["image_size"])
    )
    log.info("Loading val dataset...")
    val_ds = RoofCropDataset(
        CFG["val_dir"],
        get_transforms(train=False, image_size=CFG["image_size"])
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
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
    params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model parameters: {params:.1f}M")

    # Loss with class weights
    weights   = torch.tensor(CFG["class_weights"], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = Adam(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CFG["epochs"],
        eta_min=1e-6,
    )

    mlflow.set_experiment(CFG["mlflow_experiment"])

    best_acc   = 0.0
    no_improve = 0
    best_path  = Path(CFG["output_dir"]) / "best_efficientnet.pth"

    with mlflow.start_run(run_name="efficientnetv2-s-run1"):
        mlflow.log_params(CFG)

        for epoch in range(1, CFG["epochs"] + 1):
            t0 = time.time()

            tr_loss, tr_acc, tr_cls = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            va_loss, va_acc, va_cls = validate(
                model, val_loader, criterion, device
            )
            scheduler.step()
            elapsed = time.time() - t0

            log.info(
                f"Epoch {epoch:3d}/{CFG['epochs']} | "
                f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                f"Val loss={va_loss:.4f} acc={va_acc:.4f} | "
                f"{elapsed:.0f}s"
            )

            # Per-class metrics
            for cls_name, m in va_cls.items():
                log.info(
                    f"  {cls_name:8s}: "
                    f"P={m['precision']:.3f} "
                    f"R={m['recall']:.3f} "
                    f"F1={m['f1']:.3f}"
                )

            mlflow.log_metrics({
                "train_loss": tr_loss,
                "train_acc":  tr_acc,
                "val_loss":   va_loss,
                "val_acc":    va_acc,
            }, step=epoch)

            # Per class F1 to MLflow
            for cls_name, m in va_cls.items():
                mlflow.log_metric(
                    f"val_f1_{cls_name}", m["f1"], step=epoch
                )

            # Save best
            if va_acc > best_acc:
                best_acc   = va_acc
                no_improve = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_acc":     va_acc,
                    "val_metrics": va_cls,
                    "cfg":         CFG,
                }, best_path)
                log.info(f"  ✓ New best accuracy: {best_acc:.4f} → saved")
            else:
                no_improve += 1
                log.info(
                    f"  No improvement "
                    f"({no_improve}/{CFG['early_stop']})"
                )

            if no_improve >= CFG["early_stop"]:
                log.info(f"Early stopping at epoch {epoch}")
                break

        # Final per-class report
        log.info("\n" + "=" * 50)
        log.info("FINAL VALIDATION METRICS:")
        log.info(f"Best accuracy: {best_acc:.4f}")
        _, final_acc, final_cls = validate(
            model, val_loader, criterion, device
        )
        for cls_name, m in final_cls.items():
            log.info(
                f"  {cls_name:8s}: "
                f"Precision={m['precision']:.4f}  "
                f"Recall={m['recall']:.4f}  "
                f"F1={m['f1']:.4f}"
            )
        log.info("=" * 50)
        mlflow.log_artifact(str(best_path))


if __name__ == "__main__":
    main()
