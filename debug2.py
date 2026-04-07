import torch
from dataset import get_dataloader
from train import build_model

# Load one single sample
loader, dataset = get_dataloader("data/", batch_size=1)
images, targets = next(iter(loader))

device = torch.device("cpu")
model = build_model(num_classes=2)
model.to(device)
model.train()

images = [img.to(device) for img in images]
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

print("Image shape:", images[0].shape)
print("Boxes:", targets[0]['boxes'])
print("Labels:", targets[0]['labels'])

# Run forward pass and print each loss
loss_dict = model(images, targets)
print("\nLoss components:")
for k, v in loss_dict.items():
    print(f"  {k}: {v.item()}")