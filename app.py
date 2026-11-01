import streamlit as st
import torch
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Define Generator class to load the model
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(100, 512, 4, 1, 0),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
generator.load_state_dict(torch.load('models/generator.pth', map_location=device))
generator.eval()

st.title("Human Face Generation with DCGAN")

if st.button("Generate a Face"):
    noise = torch.randn(1, 100, 1, 1, device=device)
    with torch.no_grad():
        generated_img = generator(noise)
    grid = make_grid(generated_img, nrow=1, normalize=True)
    np_img = grid.permute(1, 2, 0).cpu().numpy()
    st.image(np_img, caption="Generated Human Face", use_column_width=True)
