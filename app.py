import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np

# Load model architecture
class VAE(torch.nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 400)
        self.fc21 = torch.nn.Linear(400, latent_dim)
        self.fc22 = torch.nn.Linear(400, latent_dim)
        self.fc3 = torch.nn.Linear(latent_dim, 400)
        self.fc4 = torch.nn.Linear(400, 28*28)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

# Load trained model
device = torch.device("cpu")
model = VAE()
model.load_state_dict(torch.load("vae_mnist.pth", map_location=device))
model.eval()

# UI
st.title("MNIST Handwritten Digit Generator")
digit = st.selectbox("Select a digit to generate", list(range(10)))
generate = st.button("Generate")

if generate:
    # Load MNIST to get latent space for the selected digit
    transform = transforms.ToTensor()
    dataset = MNIST(root='./data', train=True, download=True, transform=transform)

    # Get all examples of the selected digit
    images = [img for img, label in dataset if label == digit]
    images = torch.stack(images[:100])  # limit to first 100 of that digit
    images = images.view(-1, 28*28)

    with torch.no_grad():
        mu, logvar = model.encode(images)
        z = model.reparameterize(mu, logvar)

        # Randomly sample 5 encodings and decode them
        selected = z[torch.randint(0, z.size(0), (5,))]
        generated = model.decode(selected).view(-1, 28, 28)

    # Display images
    st.subheader(f"Generated images of digit: {digit}")
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axs):
        ax.imshow(generated[i].numpy(), cmap='gray')
        ax.axis('off')
    st.pyplot(fig)
