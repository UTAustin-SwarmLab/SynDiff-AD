import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from pytorch_models.sigma_vae import Flatten


class DetectionNetwork(nn.Module):
    def __init__(self, device="cuda", img_channels=1, out_dim=16, args=None):
        super().__init__()
        self.device = device
        self.out_dim = out_dim
        self.img_channels = img_channels

        img_size = 100
        filters_m = 32

        ## Build network
        self.detector = self.get_detector_network(self.img_channels, filters_m)

        # output size depends on input image size, compute the output size
        demo_input = torch.ones([1, self.img_channels, img_size, img_size])
        h_dim = self.detector(demo_input).shape[1]
        self.fc = nn.Linear(h_dim, self.out_dim)

        self.mse_loss = nn.MSELoss()

    def get_detector_network(self, img_channels, filters_m):
        return nn.Sequential(
            nn.Conv2d(img_channels, filters_m, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters_m, 2 * filters_m, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * filters_m, 4 * filters_m, (5, 5), stride=2, padding=2),
            nn.ReLU(),
            Flatten(),
        )

    def forward(self, x):
        return self.fc(self.detector(x))

    def loss(self, y, y_hat):
        return self.mse_loss(y, y_hat)
