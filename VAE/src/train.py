import torch
import torch.nn as nn
from torch.optim import Adam
from model import VAE
from data_loader import get_dataloaders
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

def loss_function_VAE(recon_x, x, mu, logvar, beta):
    bce_loss = nn.BCELoss(reduction='sum')
    recon_loss = bce_loss(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_divergence
    return total_loss, recon_loss, kl_divergence

def train(dataloader, model, loss_fn, optimizer, beta):
    model.train()
    total_loss, recon_loss, kl_loss = 0, 0, 0
    for batch, (x, _) in enumerate(dataloader):
        x = x.to(device)
        recon_x, mu, logvar = model(x)
        loss, recon, kl = loss_fn(recon_x, x, mu, logvar, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        recon_loss += recon.item()
        kl_loss += kl.item()
        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{batch * len(x):>5d}/{len(dataloader.dataset):>5d}]")
    return total_loss / len(dataloader.dataset), recon_loss / len(dataloader.dataset), kl_loss / len(dataloader.dataset)

def main():
    train_loader, val_loader, test_loader = get_dataloaders()
    latent_dim = 10
    model = VAE(latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    beta = 1.0
    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss, train_recon_loss, train_kl_loss = train(train_loader, model, loss_function_VAE, optimizer, beta)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Recon Loss: {train_recon_loss:.4f}, KL Loss: {train_kl_loss:.4f}")

    torch.save(model.state_dict(), 'vae.pth')

if __name__ == '__main__':
    main()
