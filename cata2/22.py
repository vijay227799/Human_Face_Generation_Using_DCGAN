import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the enhanced model structure (to output both percentage and future prediction)
class CataractModel(nn.Module):
    def _init_(self, num_classes=2):
        super(CataractModel, self)._init_()
        self.resnet = models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        
        # Replace the final layer with two outputs:
        # 1. For cataract percentage (regression or classification into ranges)
        # 2. For cataract development prediction (binary classification)
        self.resnet.fc = nn.Linear(num_ftrs, 2)  # First for percentage, second for future prediction
    
    def forward(self, x):
        return self.resnet(x)

# Instantiate the model
model_ft = CataractModel()
model_ft = model_ft.to(device)

# Load the saved model weights
model_ft.load_state_dict(torch.load('your_model.pth'))
model_ft.eval()  # Set the model to evaluation mode

# Define the same transformations used for training
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to predict cataract percentage and future development
def predict_image(image_path, model):
    image = Image.open(image_path)
    image = data_transforms(image).unsqueeze(0)  # Add batch dimension

    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        
        # Output 0: Cataract severity as percentage
        cataract_percentage = outputs[0].item() * 100  # Assuming normalized output in range [0, 1]
        
        # Output 1: Binary future cataract development prediction (0 or 1)
        future_cataract = torch.sigmoid(outputs[1]).item()  # Output probability
        
        # Interpretation of results
        future_cataract_label = "Yes" if future_cataract > 0.5 else "No"
        
    return cataract_percentage, future_cataract_label
"""
# Example usage
image_path = 'path_to_new_image.jpg'  # Replace with the path to the new image
cataract_percentage, future_cataract_label = predict_image(image_path, model_ft)

print(f'Cataract Severity: {cataract_percentage:.2f}%')
print(f'Will Develop Cataract in Future: {future_cataract_label}')
"""