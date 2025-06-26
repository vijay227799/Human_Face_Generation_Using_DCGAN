import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os

# Paths to your dataset
data_dir = 'processed_images'
val_dir = os.path.join(data_dir, 'val')

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load validation data
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: cataract, normal
model.load_state_dict(torch.load('your_model.pth'))
model.to(device)

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Evaluate on validation set
val_accuracy = evaluate_model(model, val_loader, device)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
