import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh()
        )
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x.view(-1, 784))

def train_gan(epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    criterion = nn.BCELoss()
    optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(loader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Train Discriminator
            netD.zero_grad()
            label_real = torch.ones(batch_size, 1).to(device)
            output = netD(real_imgs)
            lossD_real = criterion(output, label_real)
            
            noise = torch.randn(batch_size, 100).to(device)
            fake_imgs = netG(noise)
            label_fake = torch.zeros(batch_size, 1).to(device)
            output = netD(fake_imgs.detach())
            lossD_fake = criterion(output, label_fake)
            (lossD_real + lossD_fake).backward()
            optD.step()

            # Train Generator
            netG.zero_grad()
            output = netD(fake_imgs)
            lossG = criterion(output, label_real)
            lossG.backward()
            optG.step()
            
        print(f"GAN Epoch {epoch+1} | LossD: {lossD_real+lossD_fake:.4f} LossG: {lossG:.4f}")

    torch.save(netG.state_dict(), "gan_mnist.pth")
    print("GAN Checkpoint saved.")

if __name__ == "__main__":
    train_gan()