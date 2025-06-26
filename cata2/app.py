import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Define the model architecture
class CataractModel(nn.Module):
    def _init_(self):
        super(CataractModel, self)._init_()
        self.resnet = models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        return self.resnet(x)

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

def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        cataract_prob = probs[0][1].item()
        future_cataract_prob = probs[0][0].item()

    # Return complementary percentage for severity
    return (1 - cataract_prob) * 100, future_cataract_prob * 100

# Streamlit frontend
st.set_page_config(page_title="Cataract Detection", page_icon=":eye:")
st.title("Cataract Detection and Future Risk Prediction")

st.write("Upload an image to check for cataract severity and the likelihood of developing cataracts in the future.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    cataract_percentage, future_cataract_percentage = predict(image)

    st.write(f"Cataract Severity: {cataract_percentage:.2f}%")
    
    if future_cataract_percentage < 50:
        st.write("Likelihood of Developing Cataracts in the Future: Yes")
    elif future_cataract_percentage == 0.00:
        st.write("You don't have cataract currently")
    else:
        st.write("The uploaded image already has cataract. Danger!!!!")
else:
    st.write("Upload an image to see predictions.")

# Custom CSS to style the Streamlit app
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #003366;  /* Dark blue background */
    }
    .reportview-container .markdown-text-container {
        color: #FFFFFF;  /* White text color */
    }
    .streamlit-expanderHeader {
        color: #FFFFFF;  /* White text color for expander headers */
    }
    .css-1g7b2u4 {
        color: #FFFFFF;  /* White text color for widgets */
    }
    </style>
    """,
    unsafe_allow_html=True
)