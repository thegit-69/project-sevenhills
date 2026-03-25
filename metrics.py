import torch

# 1. Check SegFormer (The Building Extractor)
seg = torch.load("/home/kalki/models/segformer_binary/best_segformer.pth", map_location="cpu")
print("\n--- SEGFORMER ---")
print(f"Saved at Epoch: {seg['epoch']}")
print(f"Best mIoU: {seg['val_miou']:.4f}")
print(f"Class Scores: {seg['val_cls']}")

# 2. Check DeepLab (The Infrastructure Expert)
dl = torch.load("/home/kalki/models/deeplabv3/best_deeplabv3.pth", map_location="cpu")
print("\n--- DEEPLAB ---")
print(f"Saved at Epoch: {dl['epoch']}")
print(f"Best mIoU: {dl['val_miou']:.4f}")
print(f"Class Scores: {dl['val_cls']}")

exit()
