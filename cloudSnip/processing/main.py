import torch
from PIL import Image
from torchvision import transforms

# Load and preprocess image
img = Image.open("bus.jpg").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [0, 1] float tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

input_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

# Load model
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

print(dino)

# Inference
with torch.no_grad():
    output = dinov2_vits14(input_tensor)



print(output.shape)  # [1, 384] for ViT-S/14
