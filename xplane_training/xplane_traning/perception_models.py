import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import os
os.chdir("/home/pulkit/diffusion-model-based-task-driven-training/xplane_training/xplane_traning")
from xplane_traning.sigma_vae import ConvVAE

class PerceptionWaypointModel(nn.Module):
    def __init__(
        self, vae: ConvVAE, input_size: int, output_size: int, device: str, vae_save_path, waypoint_len, small=False, check_frozen=True
    ):
        super().__init__()

        self.small = small

        if input_size == output_size:
            self.fc1 = nn.Linear(input_size, input_size)
            self.fc2 = nn.Linear(input_size, input_size)
            self.fc3 = nn.Linear(input_size, output_size)
        elif small and input_size == 128 + 8:
            self.fc1 = nn.Linear(136, 64)
            self.fc2 = nn.Linear(64, output_size)
        elif input_size == 128 + 8:
            self.fc1 = nn.Linear(136, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, output_size)
        else:
            self.fc1 = nn.Linear(input_size, input_size - 5)
            self.fc2 = nn.Linear(input_size - 5, input_size - 15)
            self.fc3 = nn.Linear(input_size - 15, output_size)

        self.mse_loss = nn.MSELoss()
        self.vae = vae

        if check_frozen:
            for parameter in self.vae.parameters():
                assert not parameter.requires_grad

        self.waypoint_len = waypoint_len
        self.device = device

    def apply_model(self, z, history):
        if history is not None:
            x = torch.cat((z, history), 1)
        else:
            x = z

        if self.small:
            x = F.relu(self.fc1(x))
            y_hat = self.fc2(x)
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            y_hat = self.fc3(x)
        return y_hat

    def adversarial_forward(self, z, history):
        recon_scene = self.vae.decode(z)
        z, _ = self.vae.encode(recon_scene)
        y_hat = self.apply_model(z, history)
        return y_hat, recon_scene

    def forward(self, scene, history):
        if len(scene.shape) == 3:
            scene = scene.unsqueeze(1)
        z, _ = self.vae.encode(scene)
        y_hat = self.apply_model(z, history)
        return y_hat

    def check_ellipsoid_collision(self, wp, obstacles, path_len):
        wp = wp.reshape(-1, path_len, 2)
        obstacles = obstacles.unsqueeze(2)
        obstacles = obstacles.repeat(1, 1, path_len, 1)
        num_collisions = torch.zeros((wp.shape[0])).to(self.device)
        for idx in range(obstacles.shape[1]):
            p = torch.div(
                torch.square(wp[:, :, 0] - obstacles[:, idx, :, 0]), torch.square(obstacles[:, idx, :, 2])
            ) + torch.div(torch.square(wp[:, :, 1] - obstacles[:, idx, :, 1]), torch.square(obstacles[:, idx, :, 3]))
            p = p <= 1
            num_collisions += p.sum(1)
        return num_collisions

    def loss(self, y, y_hat, obstacle, collision_gain=100.0, use_collision=True):
        total_loss = self.mse_loss(y, y_hat)

        num_collisions = 0
        if use_collision:
            num_collisions = self.check_ellipsoid_collision(y_hat, obstacle, self.waypoint_len)
            total_loss += collision_gain * num_collisions.sum()

        return total_loss, num_collisions
