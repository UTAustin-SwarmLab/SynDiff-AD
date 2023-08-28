"""
    TINY TAXINET training script
    Code to train a DNN vision model to predict aircraft state variables
    REQUIRES:
        - raw camera images (training) in DATA_DIR + '/nominal_conditions'
        - validation data in DATA_DIR + '/nominal_conditions_val/'

    FUNCTIONALITY:
        - DNN is a small custom DNN with learnable final linear layer
            for regression with N=2 outputs
        - N=2 state outputs are:
            - distance_to_centerline
            - downtrack_position
        - trains for configurable number of epochs
        - saves the best model params and loss plot in 
            - SCRATCH_DIR + '/tiny_taxinet_DNN_train/'

"""
DEVICE_NAME = "pulkit"
# frac = "0.2"    
import time
import copy
import os
import sys
import torch
import numpy as np
from tqdm.autonotebook import tqdm as tqdm
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from colorama import Fore

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from taxinet_dataloader import (
    # tiny_taxinet_prepare_dataloader,
    taxinet_prepare_dataloader,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    transform_denormalize,
    ExprType,
)
from xplane_training.plot_utils import basic_plot_ts
from xplane_training.utils import SEED, remove_and_create_dir
SEED = 0

from xplane_traning.soft_vae import SoftIntroVAE, calc_kl, calc_reconstruction_loss

# make sure this is a system variable in your bashrc
# NASA_ULI_ROOT_DIR = os.environ["NASA_ULI_ROOT_DIR"]

# DATA_DIR = os.environ["NASA_DATA_DIR"]

# where intermediate results are saved
# never save this to the main git repo
# SCRATCH_DIR = NASA_ULI_ROOT_DIR + "/scratch/"

torch.manual_seed(SEED)
BETA_REC = 0.2
BETA_KL = 0.2
Z_DIM = 128


def train_model(
    model,
    datasets,
    dataloaders,
    optimizers,
    device,
    results_dir,
    num_epochs=25,
    log_every=100,
):
    """
    Trains a model on datatsets['train'] using criterion(model(inputs), labels) as the loss.
    Returns the model with lowest loss on datasets['val']
    Puts model and inputs on device.
    Trains for num_epochs passes through both datasets.

    Writes tensorboard info to ./runs/ if given
    """
    writer = None
    writer = SummaryWriter(log_dir=results_dir)

    model = torch.nn.DataParallel(model)
    model = model.to(device)

    since = time.time()

    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val"]}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    tr_loss = np.nan
    val_loss = np.nan

    n_tr_batches_seen = 0

    train_loss_vec = []
    val_loss_vec = []

    optimizer_e, e_scheduler = optimizers[0]
    optimizer_d, d_scheduler = optimizers[1]

    with tqdm(total=num_epochs, position=0) as pbar:
        pbar.bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)

        pbar2 = tqdm(total=dataset_sizes["train"], position=1)
        pbar2.bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)

        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0

                running_tr_loss = 0.0  # used for logging
                running_recon_loss = 0.0
                running_kl_loss = 0.0

                running_n = 0

                # Iterate over data.
                pbar2.refresh()
                pbar2.reset(total=dataset_sizes[phase])

                for data in dataloaders[phase]:
                    if phase == "train":
                        n_tr_batches_seen += 1

                    inputs, _, _, _ = data
                    inputs = inputs.to(device)
                    # print(outputs.shape)
                    if len(inputs.shape) == 3:
                        inputs = inputs.unsqueeze(1)
                        # print(inputs)

                    # zero the parameter gradients
                    optimizer_d.zero_grad()
                    optimizer_e.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):

                        # Run VAE
                        real_mu, real_logvar, z, recon_batch = model(inputs)

                        # Compute loss
                        loss_rec = calc_reconstruction_loss(
                            inputs,
                            recon_batch,
                            loss_type="mse",
                            reduction="mean",
                        )
                        loss_kl = calc_kl(real_logvar, real_mu, reduce="mean")

                        loss = BETA_REC * loss_rec + BETA_KL * loss_kl

                        ####################

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer_e.step()
                            optimizer_d.step()

                    # statistics
                    running_loss += loss.item() * inputs.shape[0]

                    if phase == "train":
                        running_n += inputs.shape[0]
                        running_tr_loss += loss.item() * inputs.shape[0]
                        running_recon_loss += loss_rec.item() * inputs.shape[0]
                        running_kl_loss += loss_kl.item() * inputs.shape[0]

                        if n_tr_batches_seen % log_every == 0:
                            mean_loss = running_tr_loss / running_n
                            mean_recon_mse = running_recon_loss / running_n
                            mean_kl = running_kl_loss / running_n

                            writer.add_scalar("loss/train", mean_loss, n_tr_batches_seen)
                            writer.add_scalar("loss/recon", mean_recon_mse, n_tr_batches_seen)
                            writer.add_scalar("loss/kl", mean_kl, n_tr_batches_seen)

                            running_tr_loss = 0.0
                            running_n = 0
                            running_recon_loss = 0.0
                            running_kl_loss = 0.0

                    pbar2.set_postfix(
                        split=phase,
                        batch_loss=loss.item(),
                        recon=loss_rec.item(),
                        kl=loss_kl.item(),
                    )
                    pbar2.update(inputs.shape[0])

                epoch_loss = running_loss / dataset_sizes[phase]

                if phase == "train":
                    tr_loss = epoch_loss
                    train_loss_vec.append(tr_loss)

                    if epoch % log_every == 0:
                        torch.save(
                            best_model_wts,
                            results_dir + f"/best_model_{epoch}.pt",
                        )

                if phase == "val":
                    val_loss = epoch_loss
                    writer.add_scalar("loss/val", val_loss, n_tr_batches_seen)
                    val_loss_vec.append(val_loss)

                    # Sampling latent space
                    z = torch.randn(64, Z_DIM).to(device)
                    sample = model.module.sample(z).cpu()
                    save_image(
                        transform_denormalize(sample.view(64, -1, IMAGE_HEIGHT, IMAGE_WIDTH)),
                        results_dir + "output_images/sample_{}.png".format(str(epoch)),
                    )

                    # Recon comparison
                    n = min(recon_batch.size(0), 8)
                    comparison = torch.cat(
                        [
                            transform_denormalize(inputs[:n]),
                            transform_denormalize(
                                recon_batch.view(recon_batch.shape[0], -1, IMAGE_HEIGHT, IMAGE_WIDTH)[:n]
                            ),
                        ]
                    )
                    save_image(
                        comparison.cpu(),
                        results_dir + "output_images/recon_{}.png".format(str(epoch)),
                        nrow=n,
                    )

                pbar.set_postfix(tr_loss=tr_loss, val_loss=val_loss)

                # deep copy the model
                if phase == "val" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                # best_model_wts = copy.deepcopy(model.state_dict())

            pbar.update(1)

            print(" ")
            print("training loss: ", train_loss_vec)
            print("val loss: ", val_loss_vec)
            print(" ")

        e_scheduler.step()
        d_scheduler.step()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Loss: {:4f}".format(best_loss))

    writer.flush()

    # plot the results to a file
    plot_file = results_dir + "/loss.pdf"
    basic_plot_ts(train_loss_vec, val_loss_vec, plot_file, legend=["Train Loss", "Val Loss"])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_options = {
        "epochs": 100,
        "learning_rate": 1e-3,
    }

    dataloader_params = {
        "batch_size": 1200,
        "shuffle": True,
        "num_workers": 64,
        "drop_last": False,
        "pin_memory": False,
    }

    print("found device: ", device)

    # condition_list = ['afternoon']

    # experiment_list = [['afternoon'], ['morning'], ['overcast'], ['night'], ['afternoon', 'morning', 'overcast', 'night']]
    for frac in ["0.2","0.3","0.4"]:
        experiment_list = [["afternoon"]]
        SCRATCH_DIR = "/home/pulkit/datasets_xplane/0.2/trained_models_svae".replace("pulkit", DEVICE_NAME).replace("0.2", frac)

        for condition_list in experiment_list:

            condition_str = "_".join(condition_list)
            print(condition_str)

            HISTORY_LEN = 4
            # where the training results should go
            results_dir = remove_and_create_dir(SCRATCH_DIR + f"/taxinet_VAE_Z{Z_DIM}_all_data/" + condition_str + "/")

            data_dir = "/home/pulkit/datasets_xplane/0.2/xplane_dataset/".replace("pulkit", DEVICE_NAME).replace("0.2", frac)
            train_dataset, train_loader = taxinet_prepare_dataloader(
                data_dir,
                condition_str,
                "train",
                dataloader_params,
                HISTORY_LEN,
                None,
                frac,
                [ExprType.standard],
                # [ExprType.orig],
            )

            val_dataset, val_loader = taxinet_prepare_dataloader(
                data_dir,
                condition_str,
                "val",
                dataloader_params,
                HISTORY_LEN,
                train_dataset.scaler,
                frac,
                [ExprType.standard],
                # [ExprType.orig],
            )

            assert IMAGE_HEIGHT == IMAGE_WIDTH

            model = SoftIntroVAE(
                cdim=3,
                zdim=Z_DIM,
                channels=[64, 128, 256, 512, 512],
                image_size=IMAGE_HEIGHT,
            ).to(device)

            # weights = torch.load(results_dir + "", map_location=device)
            # model.load_state_dict(weights["model"], strict=False)

            os.makedirs(data_dir + "figures", exist_ok=True)
            # OPTIMIZER
            lr_e = 2e-4
            lr_d = 2e-4
            optimizer_e = torch.optim.Adam(model.encoder.parameters(), lr=lr_e)
            optimizer_d = torch.optim.Adam(model.decoder.parameters(), lr=lr_d)

            e_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=(350,), gamma=0.1)
            d_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(350,), gamma=0.1)
            scale = 1 / (3 * IMAGE_WIDTH ** 2)  # normalize by images size (channels * height * width)

            # DATASET INFO
            datasets = {}
            datasets["train"] = train_dataset
            datasets["val"] = val_dataset

            dataloaders = {}
            dataloaders["train"] = train_loader
            dataloaders["val"] = val_loader

            # train the DNN
            model = train_model(
                model,
                datasets,
                dataloaders,
                [(optimizer_e, e_scheduler), (optimizer_d, d_scheduler)],
                device,
                results_dir,
                num_epochs=train_options["epochs"],
                log_every=5,
            )
            # results_dir = "/home/pulkit/datasets_xplane/0.2/results_dir".replace("pulkit", DEVICE_NAME).replace("0.2", frac)
            # save the best model to the directory
            torch.save(model.state_dict(), results_dir + "/best_model_svae.pt")
