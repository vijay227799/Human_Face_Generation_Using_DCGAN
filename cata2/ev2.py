import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Define the model architecture (same as used in training)
class CataractModel(nn.Module):
    def __init__(self):
        super(CataractModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        return self.resnet(x)

# Function to load the model
def load_model(model_path):
    model = CataractModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Path to the saved model
model_path = 'cataract_model.pth'
model = load_model(model_path)

# Define the same transformations used for validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function for predicting
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        cataract_prob = probs[0][1].item()
        future_cataract_prob = probs[0][0].item()

    return cataract_prob * 100, future_cataract_prob * 100

# Test the prediction function
image_path = 'C:\\cata2\\processed_images\\test\\normal1\\image_253.png'  # Path to the image you want to test
cataract_percentage, future_cataract_percentage = predict(image_path)

# Output results
print(f"Cataract Severity: {cataract_percentage:.2f}%")
print(f"Future Cataract Development Risk: {future_cataract_percentage:.2f}%")
