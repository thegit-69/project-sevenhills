import torch

# SegFormer
ckpt = torch.load('/home/kalki/models/segformer_binary/best_segformer.pth', map_location='cpu', weights_only=False)
print('=== SEGFORMER ===')
print('Saved at Epoch:', ckpt['epoch'])
print('Best mIoU:', ckpt['val_miou'])
print('Class Scores:', ckpt['val_cls'])

# DeepLab
ckpt = torch.load('/home/kalki/models/deeplab/best_deeplabv3.pth', map_location='cpu', weights_only=False)
print('=== DEEPLAB ===')
print('Saved at Epoch:', ckpt['epoch'])
print('Best mIoU:', ckpt['val_miou'])
print('Class Scores:', ckpt['val_cls'])

# EfficientNet
ckpt = torch.load('/home/kalki/models/efficientnet/best_efficientnet.pth', map_location='cpu', weights_only=False)
print('=== EFFICIENTNET ===')
print('Saved at Epoch:', ckpt['epoch'])
print('Best val acc:', ckpt['val_acc'])
print('Class Metrics:', ckpt['val_metrics'])