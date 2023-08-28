import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


def softclip(tensor, min):
    """Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials"""
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels, final_enc_conv_shape):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels
        self.con_x = final_enc_conv_shape[0]
        self.con_y = final_enc_conv_shape[1]

    def forward(self, input):
        return input.view(input.size(0), self.n_channels, self.con_x, self.con_y)


class ConvVAE(nn.Module):
    def __init__(
        self,
        device="cuda",
        img_channels=3,
        img_h=100,
        img_w=100,
        loss_type="optimal_sigma_vae",
        z_dim=20,
        filters_m=32,
    ):
        super().__init__()
        self.device = device
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.model = loss_type
        filters_m = 32

        ## Build network
        self.encoder = self.get_encoder(self.img_channels, filters_m)

        # output size depends on input image size, compute the output size
        demo_input = torch.ones([1, self.img_channels, img_h, img_w])
        encoder_out = self.encoder(demo_input)
        self.final_conv_shapes = encoder_out.shape[-2:]
        h_dim = Flatten()(encoder_out).shape[1]
        print("h_dim", h_dim)

        # map to latent z
        self.fc11 = nn.Linear(h_dim, self.z_dim)
        self.fc12 = nn.Linear(h_dim, self.z_dim)

        # decoder
        self.fc2 = nn.Linear(self.z_dim, h_dim)
        self.decoder = self.get_decoder(filters_m, self.img_channels)

        ## Build network
        self.encoder = self.get_encoder(self.img_channels, filters_m)

        self.log_sigma = 0
        if self.model == "sigma_vae":
            ## Sigma VAE
            param = torch.full((1,), 0)[0].type(torch.FloatTensor)
            self.log_sigma = torch.nn.Parameter(param, requires_grad=True)

    def get_encoder(self, img_channels, filters_m):
        return nn.Sequential(
            nn.Conv2d(img_channels, filters_m, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters_m, 2 * filters_m, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * filters_m, 4 * filters_m, (5, 5), stride=2, padding=2),
            nn.ReLU(),
        )

    def get_decoder(self, filters_m, out_channels):
        return nn.Sequential(
            UnFlatten(4 * filters_m, self.final_conv_shapes),
            nn.ConvTranspose2d(4 * filters_m, 2 * filters_m, (6, 6), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * filters_m, filters_m, (6, 6), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(filters_m, out_channels, (5, 5), stride=1, padding=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(self.fc2(z))

    def encode(self, x):
        h = Flatten()(self.encoder(x))
        return self.fc11(h), self.fc12(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, n):
        sample = torch.randn(n, self.z_dim).to(self.device)
        return self.decode(sample)

    def reconstruction_loss(self, x_hat, x):
        """Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1"""

        if self.model == "gaussian_vae":
            # Naive gaussian VAE uses a constant variance
            log_sigma = torch.zeros([], device=x_hat.device)
        elif self.model == "sigma_vae":
            # Sigma VAE learns the variance of the decoder as another parameter
            log_sigma = self.log_sigma
        elif self.model == "optimal_sigma_vae":
            log_sigma = ((x - x_hat) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()
            self.log_sigma = log_sigma.item()
        else:
            raise NotImplementedError

        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(log_sigma, -6)

        rec = gaussian_nll(x_hat, log_sigma, x).sum()

        return rec

    def loss_function(self, recon_x, x, mu, logvar):
        # Important: both reconstruction and KL divergence loss have to be summed over all element!
        # Here we also sum the over batch and divide by the number of elements in the data later
        if self.model == "mse_vae":
            rec = torch.nn.MSELoss()(recon_x, x)
        else:
            rec = self.reconstruction_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return rec, kl


def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)
