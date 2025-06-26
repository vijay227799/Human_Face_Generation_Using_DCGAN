import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
from sklearn.model_selection import KFold

# Paths to your dataset
data_dir = 'processed_images'

# Data transforms with augmentation for the training set
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Use basic transforms for validation (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
train_dir = os.path.join(data_dir, 'train')
dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)

# Define model, loss function, and optimizer
class CataractModel(nn.Module):
    def __init__(self):
        super(CataractModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        # Replace the final layer with two outputs:
        # 1. For cataract severity percentage (regression)
        # 2. For future cataract development prediction (binary classification)
        self.resnet.fc = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        return self.resnet(x)

model = CataractModel()

criterion = nn.CrossEntropyLoss()  # For binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# K-fold cross-validation
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

# Train the model with cross-validation
num_epochs = 5
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f'FOLD {fold+1}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids for train/val split
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # DataLoaders for training and validation
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_subsampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=val_subsampler)

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

        # Validation loop (optional: you can add this to monitor overfitting)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {accuracy:.2f}%')

    print('--------------------------------')

# Save the model after training all folds
torch.save(model.state_dict(), 'cataract_model.pth')