import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.enc_conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 256 * 2 * 2)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, output_padding=0)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.relu(self.enc_conv1(x))
        x = self.relu(self.enc_conv2(x))
        x = self.relu(self.enc_conv3(x))
        x = self.relu(self.enc_conv4(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.relu(self.dec_fc(z))
        z = z.view(-1, 256, 2, 2)
        z = self.relu(self.dec_conv1(z))
        z = self.relu(self.dec_conv2(z))
        z = self.relu(self.dec_conv3(z))
        z = self.sigmoid(self.dec_conv4(z))
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
