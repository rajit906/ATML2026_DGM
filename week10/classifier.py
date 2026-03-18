import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset

# 1. Define the model class here so it can be imported elsewhere
class TimeClassifier(nn.Module):
    def __init__(self, num_steps=100, hidden_dim=128):
        super().__init__()
        self.num_steps = num_steps
        self.net = nn.Sequential(
            nn.Linear(2 + 1, hidden_dim), 
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, z, t):
        t_norm = (t.float() / self.num_steps).view(-1, 1)
        x_input = torch.cat([z, t_norm], dim=1)
        return self.net(x_input)

# 2. Training script executes only if run directly
if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Minimal recreation of the noise schedule for training
    num_steps = 100
    betas = torch.linspace(1e-4, 0.02, num_steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    def q_sample(x_0, t, noise=None):
        if noise is None: noise = torch.randn_like(x_0)
        alpha_cumprod_t = alphas_cumprod[t].view(-1, 1)
        return torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1 - alpha_cumprod_t) * noise

    # Load data with labels
    X_moons, y_moons = make_moons(n_samples=10000, noise=0.05, random_state=SEED)
    X_moons = torch.tensor(X_moons, dtype=torch.float32)
    y_moons = torch.tensor(y_moons, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(X_moons, y_moons), batch_size=256, shuffle=True)

    # Initialize and train
    classifier = TimeClassifier(num_steps=num_steps).to(device)
    opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    print("Training Time-Dependent Classifier...")
    for epoch in range(50):
        for x_0, y in dataloader:
            x_0, y = x_0.to(device), y.to(device).view(-1, 1)
            t = torch.randint(0, num_steps, (x_0.shape[0],), device=device).long()
            z_t = q_sample(x_0, t)
            
            loss = F.binary_cross_entropy_with_logits(classifier(z_t, t), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save(classifier.state_dict(), "classifier.pt")
    print("Saved classifier to classifier.pt")