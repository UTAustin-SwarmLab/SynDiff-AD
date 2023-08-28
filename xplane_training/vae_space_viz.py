import os
import torch
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from xplane.taxinet_dataloader import (
    # tiny_taxinet_prepare_dataloader,
    taxinet_prepare_dataloader,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    transform_denormalize,
    condition_dict,
    ExprType,
)
from xplane.utils import SEED, remove_and_create_dir

from pytorch_models.soft_vae import SoftIntroVAE

# import torch.multiprocessing

# torch.multiprocessing.set_sharing_strategy("file_system")


# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR = os.environ["NASA_ULI_ROOT_DIR"]

DATA_DIR = os.environ["NASA_DATA_DIR"]

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + "/scratch/"

torch.manual_seed(SEED)

torch.cuda.empty_cache()
device = torch.device("cpu")

train_options = {
    "epochs": 100,
    "learning_rate": 1e-3,
}

dataloader_params = {
    "batch_size": 1,
    "shuffle": True,
    "num_workers": 1,
    "drop_last": False,
    "pin_memory": True,
}

condition_str = "afternoon"
print(condition_str)

HISTORY_LEN = 4
Z_DIM = 128
NUM_SAMPLES = 1000

# where the training results should go
results_dir = remove_and_create_dir(SCRATCH_DIR + f"/taxinet_VAE_Z{Z_DIM}_train/" + condition_str + "/")

data_dir = f"xplane/data_original/{condition_str}/"

vae_model = SoftIntroVAE(
    cdim=3,
    zdim=Z_DIM,
    channels=[64, 128, 256, 512, 512],
    image_size=IMAGE_HEIGHT,
).to(device)

vae_save_path = f"/home/somi/Adversarial/xplane/scratch/" f"taxinet_VAE_Z{Z_DIM}_train/{condition_str}/best_model_45.pt"

chkpt = torch.load(vae_save_path, map_location=torch.device(device))
print(f"Using weights {vae_save_path}")
vae_model.load_state_dict(chkpt)


samples_torch = torch.zeros(NUM_SAMPLES, Z_DIM)

# train_dataset, train_loader = taxinet_prepare_dataloader(
#     data_dir,
#     condition_str,
#     "train",
#     dataloader_params,
#     HISTORY_LEN,
#     None,
#     # [ExprType.orig, ExprType.condition_morning, ExprType.condition_night],
#     [ExprType.condition_night],
# )

# idx = 0
# for data in train_loader:
#     print(idx)
#     scene, _, _, cond_label = data
#     sample, _ = vae_model.encode(scene)

#     samples_torch[idx] = sample

#     if idx == NUM_SAMPLES - 1:
#         break

#     idx += 1


# np.savez(
#     "night_samples_{}".format(NUM_SAMPLES),
#     samples_torch.detach().numpy(),
# )


morning_np = np.load("morning_samples_1000.npz")["arr_0"]
afternoon_np = np.load("afternoon_samples_1000.npz")["arr_0"]
night_np = np.load("night_samples_1000.npz")["arr_0"]

adver_sample_data = np.load("wpt_xplane_adv_4_2_5k_ADVER_samples_2500.npz")["arr_0"][:NUM_SAMPLES]


all_sample_data = np.concatenate((morning_np, afternoon_np, night_np, adver_sample_data), axis=0)
X_embedded = TSNE(n_components=3).fit_transform(all_sample_data)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(
    X_embedded[: 1 * NUM_SAMPLES, 0],
    X_embedded[: 1 * NUM_SAMPLES, 1],
    X_embedded[: 1 * NUM_SAMPLES, 2],
    c="r",
    label="morning",
)
ax.scatter3D(
    X_embedded[1 * NUM_SAMPLES : 2 * NUM_SAMPLES, 0],
    X_embedded[1 * NUM_SAMPLES : 2 * NUM_SAMPLES, 1],
    X_embedded[1 * NUM_SAMPLES : 2 * NUM_SAMPLES, 2],
    c="b",
    label="afternoon",
)
ax.scatter3D(
    X_embedded[2 * NUM_SAMPLES : 3 * NUM_SAMPLES, 0],
    X_embedded[2 * NUM_SAMPLES : 3 * NUM_SAMPLES, 1],
    X_embedded[2 * NUM_SAMPLES : 3 * NUM_SAMPLES, 2],
    c="g",
    label="night",
)
ax.scatter3D(
    X_embedded[3 * NUM_SAMPLES :, 0],
    X_embedded[3 * NUM_SAMPLES :, 1],
    X_embedded[3 * NUM_SAMPLES :, 2],
    c="y",
    label="adver",
)
ax.legend()

plt.show()
