import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from model import VAE
from data_loader import get_dataloaders
from utils import show

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

def evaluate():
    train_loader, val_loader, test_loader = get_dataloaders()
    latent_dim = 10
    model = VAE(latent_dim).to(device)
    model.load_state_dict(torch.load('vae.pth'))

    sample_inputs, _ = next(iter(test_loader))
    fixed_input = sample_inputs[0:32, :, :, :]

    img = make_grid(fixed_input, nrow=8, padding=2, normalize=False, scale_each=False, pad_value=0)
    plt.figure()
    show(img)

    with torch.no_grad():
        recon_batch, _, _ = model(sample_inputs.to(device))
        recon_batch = recon_batch.cpu()
        recon_batch = make_grid(recon_batch[0:32, :, :, :], nrow=8, padding=2, normalize=False, scale_each=False, pad_value=0)
        plt.figure()
        show(recon_batch)

if __name__ == '__main__':
    evaluate()
