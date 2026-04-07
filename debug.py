import torch
from dataset import get_dataloader

loader, dataset = get_dataloader("data/", batch_size=1)

for i, (images, targets) in enumerate(loader):
    img = images[0]
    tgt = targets[0]
    boxes = tgt['boxes']
    
    # Check for degenerate boxes (width or height <= 0)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    bad = (widths <= 0) | (heights <= 0)
    
    if bad.any():
        print(f"Frame {i}: DEGENERATE BOX FOUND")
        print(f"  Boxes: {boxes}")
        print(f"  Widths: {widths}")
        print(f"  Heights: {heights}")
    else:
        print(f"Frame {i}: OK — {len(boxes)} boxes, min_w={widths.min():.1f}, min_h={heights.min():.1f}")