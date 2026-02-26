import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc2_mu = nn.Linear(400, latent_dim)
        self.fc2_logvar = nn.Linear(400, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2_mu(h), self.fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def train_vae(epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, 
                                           transform=transforms.ToTensor()), batch_size=128, shuffle=True)

    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            # Loss = Reconstruction (MSE-like) + KL Divergence
            BCE = nn.functional.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = BCE + KLD
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"VAE Epoch {epoch+1} Loss: {train_loss / len(train_loader.dataset):.4f}")
    
    torch.save(model.state_dict(), "vae_mnist.pth")
    print("VAE Checkpoint saved.")

if __name__ == "__main__":
    train_vae()