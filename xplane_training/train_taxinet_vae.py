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

from xplane.taxinet_dataloader import (
    # tiny_taxinet_prepare_dataloader,
    taxinet_prepare_dataloader,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    transform_denormalize,
)
from xplane.plot_utils import basic_plot_ts
from xplane.utils import SEED, remove_and_create_dir

from pytorch_models.sigma_vae import ConvVAE

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR = os.environ["NASA_ULI_ROOT_DIR"]

DATA_DIR = os.environ["NASA_DATA_DIR"]

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = NASA_ULI_ROOT_DIR + "/scratch/"

torch.manual_seed(SEED)


def train_model(
    model,
    datasets,
    dataloaders,
    dist_fam,
    optimizer,
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

                running_n = 0

                # Iterate over data.
                pbar2.refresh()
                pbar2.reset(total=dataset_sizes[phase])

                for data in dataloaders[phase]:
                    if phase == "train":
                        n_tr_batches_seen += 1

                    inputs, _, _ = data
                    inputs = inputs.to(device)

                    if len(inputs.shape) == 3:
                        inputs = inputs.unsqueeze(1)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):

                        # Run VAE
                        recon_batch, mu, logvar = model(inputs)
                        # Compute loss
                        rec, kl = model.loss_function(recon_batch, inputs, mu, logvar)

                        loss = rec + kl
                        ####################

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.shape[0]

                    if phase == "train":
                        running_n += inputs.shape[0]
                        running_tr_loss += loss.item() * inputs.shape[0]

                        if n_tr_batches_seen % log_every == 0:
                            mean_loss = running_tr_loss / running_n

                            writer.add_scalar(
                                "loss/train", mean_loss, n_tr_batches_seen
                            )

                            running_tr_loss = 0.0
                            running_n = 0

                    pbar2.set_postfix(split=phase, batch_loss=loss.item())
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
                    sample = model.sample(64).cpu()
                    save_image(
                        transform_denormalize(
                            sample.view(64, -1, IMAGE_HEIGHT, IMAGE_WIDTH)
                        ),
                        results_dir + "output_images/sample_{}.png".format(str(epoch)),
                    )

                    # Recon comparison
                    n = min(recon_batch.size(0), 8)
                    comparison = torch.cat(
                        [
                            inputs[:n],
                            transform_denormalize(
                                recon_batch.view(
                                    recon_batch.shape[0], -1, IMAGE_HEIGHT, IMAGE_WIDTH
                                )[:n]
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

            pbar.update(1)

            print(" ")
            print("training loss: ", train_loss_vec)
            print("val loss: ", val_loss_vec)
            print(" ")

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Loss: {:4f}".format(best_loss))

    writer.flush()

    # plot the results to a file
    plot_file = results_dir + "/loss.pdf"
    basic_plot_ts(
        train_loss_vec, val_loss_vec, plot_file, legend=["Train Loss", "Val Loss"]
    )

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_options = {
        "epochs": 50,
        "learning_rate": 1e-3,
    }

    dataloader_params = {
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 12,
        "drop_last": False,
        "pin_memory": True,
    }

    print("found device: ", device)

    # condition_list = ['afternoon']

    # experiment_list = [['afternoon'], ['morning'], ['overcast'], ['night'], ['afternoon', 'morning', 'overcast', 'night']]
    experiment_list = [["afternoon"]]

    for condition_list in experiment_list:

        condition_str = "_".join(condition_list)
        print(condition_str)

        Z_DIM = 500
        HISTORY_LEN = 4
        # where the training results should go
        results_dir = remove_and_create_dir(
            SCRATCH_DIR + f"/taxinet_VAE_Z{Z_DIM}_train/" + condition_str + "/"
        )

        # MODEL
        # instantiate the model and freeze all but penultimate layers
        model = ConvVAE(
            device,
            img_channels=3,
            img_h=IMAGE_HEIGHT,
            img_w=IMAGE_WIDTH,
            z_dim=Z_DIM,
            filters_m=8,
        ).to(device)
        print(model)

        # DATALOADERS
        # instantiate the model and freeze all but penultimate layers
        # train_dataset, train_loader = tiny_taxinet_prepare_dataloader(
        #     DATA_DIR, condition_list, "train", dataloader_params
        # )

        # val_dataset, val_loader = tiny_taxinet_prepare_dataloader(
        #     DATA_DIR, condition_list, "validation", dataloader_params
        # )

        data_dir = f"xplane/data_original/{condition_str}/"

        train_dataset, train_loader = taxinet_prepare_dataloader(
            data_dir, condition_str, "train", dataloader_params, HISTORY_LEN, None
        )

        val_dataset, val_loader = taxinet_prepare_dataloader(
            data_dir,
            condition_str,
            "validation",
            dataloader_params,
            HISTORY_LEN,
            train_dataset.scaler,
        )

        # OPTIMIZER
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_options["learning_rate"], amsgrad=True
        )

        # LOSS FUNCTION
        loss_func = torch.nn.MSELoss().to(device)

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
            loss_func,
            optimizer,
            device,
            results_dir,
            num_epochs=train_options["epochs"],
            log_every=5,
        )

        # save the best model to the directory
        torch.save(model.state_dict(), results_dir + "/best_model.pt")
