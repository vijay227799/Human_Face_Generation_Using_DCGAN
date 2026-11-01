import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import os

# Streamlit Sidebar for Configuration
st.sidebar.title("DCGAN Configuration")
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)
image_size = st.sidebar.slider("Image Size", min_value=32, max_value=128, value=64, step=32)
learning_rate = st.sidebar.slider("Learning Rate", min_value=1e-5, max_value=1e-3, value=2e-4, step=1e-5, format="%e")
num_epochs = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=100, value=5, step=1)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Data
@st.cache_data
def load_data():
    dataset_path = 'C:/humans/dataset/Datasets/Celeba_Datasets'  # Path to the CelebA dataset folder

    # If the dataset folder does not exist, set download=True
    if not os.path.exists(dataset_path):
        st.write(f"Dataset not found at `{dataset_path}`. Attempting to download the dataset...")
        dataset_path = 'C:/humans/dataset/Datasets/Celeba_Datasets'
        download = True
    else:
        download = False

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    try:
        dataset = torchvision.datasets.CelebA(root=dataset_path, split='train', transform=transform, download=download)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader

dataloader = load_data()

if dataloader is None:
    st.stop()  # Stop execution if the dataset is not loaded

# Define Generator and Discriminator Models
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Initialize Models
generator = Generator().to(device)
discriminator = Discriminator().to(device)
opt_gen = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_disc = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Training Function
def train_dcgan():
    st.write("Training Started")
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            noise = torch.randn(batch_size, 100, 1, 1).to(device)
            fake = generator(noise)

            # Train Discriminator
            disc_real = discriminator(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = discriminator(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            output = discriminator(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Display progress
            if batch_idx % 100 == 0:
                st.write(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")
                
                # Display generated images
                with torch.no_grad():
                    fake = generator(noise)
                    img_grid_fake = make_grid(fake[:32], normalize=True)
                    st.image(img_grid_fake.permute(1, 2, 0).cpu().numpy())

if st.button('Start Training'):
    train_dcgan()
