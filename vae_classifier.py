# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import platform

# %%
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# %%
import torch
import torch.distributions
import torchvision.transforms as T
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import datasets
from torchvision.utils import make_grid


# %% [markdown]
# ## Set Device

# %%
def get_device():
    if platform.platform().lower().startswith("mac"):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    else:  # Linux, Windows
        return "cuda" if torch.cuda.is_available() else "cpu"


# %% [markdown]
# ## Encoder

# %%
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Flatten(1, -1),
        )
        self.z_mean = nn.Linear(512, 200)
        self.z_log_var = nn.Linear(512, 200)

    def forward(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var


summary(Encoder(), input_data=torch.rand((8, 1, 28, 28)))


# %% [markdown]
# ## Classifier

# %%
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(200, 64), nn.Linear(64, 10))

    def forward(self, x):
        return self.classifier(x)


# %% [markdown]
# ## Sampler

# %%
class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, x):
        # reparametriztion trick
        z_mean, z_log_var = x
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


# %% [markdown]
# ## Decoder

# %%
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(200, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, 0),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 1, 3, 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


summary(Decoder(), input_data=torch.rand((4, 200)))


# %% [markdown]
# ## VAE

# %%
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.sampler = Sampling()

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampler((z_mean, z_log_var))
        x_hat = self.decoder(z)
        return z_mean, z_log_var, x_hat


class KLDiv(nn.Module):
    def __init__(self):
        super(KLDiv, self).__init__()

    def forward(self, mean, log_var):
        # Compute the KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return kl_div / mean.size(0)


# %%
device = get_device()
vae_model = VAE().to(device)
classifier = Classifier().to(device)

# %%
optimizer = torch.optim.Adam(vae_model.parameters())
reconstruction_loss = nn.MSELoss()
classifier_loss = nn.CrossEntropyLoss()
kl_divergence = KLDiv()
epochs = 10

mnist_data = datasets.MNIST(".", train=True, download=True, transform=T.ToTensor())
train_loader = DataLoader(mnist_data, batch_size=512, shuffle=True)

# %%
loss_tracker = []
vae_model.train()
N = len(train_loader.dataset)
fixed_z = torch.randn(10, 200).to(device)
with torch.no_grad():
    x_hat = vae_model.decoder(fixed_z)
x_hat_grid = make_grid(x_hat, nrow=10)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(x_hat_grid.cpu().permute(1, 2, 0))
ax.axis("off")
plt.tight_layout()
plt.show()
for epoch in range(epochs):
    t_loss = 0
    trec = 0
    tkl = 0
    tcl = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    for image, label in pbar:
        image = image.to(device)
        label = label.to(device)
        z_mean, z_log_var, x_hat = vae_model(image)
        mean_class = classifier(z_mean)
        class_loss = classifier_loss(mean_class, label)
        rec_loss = reconstruction_loss(image, x_hat)
        kl_div = kl_divergence(z_mean, z_log_var)
        loss = 500* rec_loss + kl_div + class_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trec += rec_loss.item() * image.size(0)
        tkl += kl_div.item() * image.size(0)
        tcl += class_loss * image.size(0)
        t_loss += loss.item() * image.size(0)
        pbar.set_postfix(
            Tloss=f"{t_loss/N:.3f}",
            rec=f"{trec/N:.3f}",
            KL=f"{tkl/N:.3f}",
            tcl=f"{tcl/N:.3f}",
        )
    t_loss = t_loss / N
    loss_tracker.append(t_loss)
    with torch.no_grad():
        x_hat = vae_model.decoder(fixed_z)
    x_hat_grid = make_grid(x_hat, nrow=10)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(x_hat_grid.cpu().permute(1, 2, 0))
    ax.axis("off")
    plt.tight_layout()
    plt.show()

# %%
with torch.no_grad():
    image = image.to(device)
    label = label.to(device)
    z_mean, z_log_var, x_hat = vae_model(image)

indices = torch.randint(0, x_hat.size(0), (100,))
x_hat_subset = x_hat[indices]
x = image[indices]

x_grid = make_grid(x, nrow=10)
x_hat_grid = make_grid(x_hat_subset, nrow=10)

# Plot the grids
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(x_grid.cpu().permute(1, 2, 0))
axes[0].set_title("Original Images")
axes[0].axis("off")

axes[1].imshow(x_hat_grid.cpu().permute(1, 2, 0))
axes[1].set_title("Reconstructed Images")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# %%
with torch.no_grad():
    z = torch.randn(100, 200).to(device)
    x_hat = vae_model.decoder(z)
    indices = torch.randint(0, x_hat.size(0), (100,))
    x_hat_subset = x_hat[indices]

x_hat_grid = make_grid(x_hat_subset, nrow=10)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(x_hat_grid.cpu().permute(1, 2, 0))
ax.set_title("Sampled Images from Latent Space")
ax.axis("off")

plt.tight_layout()
plt.show()

# %%
with torch.no_grad():
    latent_range = (-10, 10)
    z = torch.rand(100, 200) * (latent_range[1] - latent_range[0]) + latent_range[0]
    z = z.to(device)
    x_hat = vae_model.decoder(z)
x_hat_grid = make_grid(x_hat, nrow=10)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.imshow(x_hat_grid.cpu().permute(1, 2, 0))
ax.set_title("Sampled Images from Latent Space (Uniform Sampling)")
ax.axis("off")

plt.tight_layout()
plt.show()

# %%
