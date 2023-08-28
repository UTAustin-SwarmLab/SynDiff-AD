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
frac = "0.2"
DEVICE_NAME = "pulkit"
import time
import copy
import os
import sys
import torch
import numpy as np
from tqdm.autonotebook import tqdm as tqdm
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from IPython import embed
from colorama import Fore

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from xplane_training.taxinet_dataloader import (
    # tiny_taxinet_prepare_dataloader,
    taxinet_prepare_dataloader,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    transform_denormalize,
    ExprType,
)
from xplane_training.plot_utils import basic_plot_ts
from xplane_training.utils import SEED, remove_and_create_dir

from xplane_training.xplane_traning.soft_vae import SoftIntroVAE
from xplane_training.xplane_traning.perception_models import PerceptionWaypointModel

# make sure this is a system variable in your bashrc
# NASA_ULI_ROOT_DIR = os.environ["NASA_ULI_ROOT_DIR"]

# DATA_DIR = os.environ["NASA_DATA_DIR"]

# # where intermediate results are saved
# # never save this to the main git repo
# SCRATCH_DIR = NASA_ULI_ROOT_DIR + "/scratch/"
SCRATCH_DIR = "/home/pulkit/datasets_xplane/".replace("pulkit", DEVICE_NAME)+frac+"/trained_models_wp"

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
    phases=["train","val"],
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

    model = torch.nn.DataParallel(model).to(device)
    since = time.time()

    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val"]}

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_loss = np.inf

    tr_loss = np.nan
    val_loss = np.nan

    n_tr_batches_seen = 0

    train_loss_vec = []
    val_loss_vec = []
    # print(model.modules)
    best_val_loss = np.inf
    with tqdm(total=num_epochs, position=0) as pbar:
        pbar.bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)

        pbar2 = tqdm(total=dataset_sizes["train"], position=1)
        pbar2.bar_format = "{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)

        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in phases:
                if phase == "train":
                    model.module.train()  # Set model to training mode
                else:
                    model.module.eval()  # Set model to evaluate mode

                running_loss = 0.0

                running_tr_loss = 0.0  # used for logging

                running_n = 0

                # Iterate over data.
                pbar2.refresh()
                pbar2.reset(total=dataset_sizes[phase])
                for data in dataloaders[phase]:
                    if phase == "train":
                        n_tr_batches_seen += 1

                    scene, history, waypoint, _ = data
                    # print(waypoint.shape)
                    scene = scene.to(device)
                    history = history.to(device)
                    waypoint = waypoint.to(device)
                    # print(waypoint.shape)
                    batch_size = scene.shape[0]

                    if len(scene.shape) == 3:
                        scene = scene.unsqueeze(1)
                    # print(scene.shape)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        
                        pred_wp = model.module(scene, history=history)
                        
                        # Compute loss
                        loss, _ = model.loss(
                            y=waypoint,
                            y_hat=pred_wp,
                            obstacle=None,
                            use_collision=False,
                   )

                        ####################

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * batch_size

                    if phase == "train":
                        running_n += batch_size
                        running_tr_loss += loss.item() * batch_size

                        if n_tr_batches_seen % log_every == 0:
                            mean_loss = running_tr_loss / running_n

                            writer.add_scalar("loss/train", mean_loss, n_tr_batches_seen)

                            running_tr_loss = 0.0
                            running_n = 0

                    pbar2.set_postfix(split=phase, batch_loss=loss.item())
                    pbar2.update(batch_size)
                
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

                pbar.set_postfix(tr_loss=tr_loss, val_loss=val_loss)

                # deep copy the model
                if phase == "val" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.module.state_dict())
                    torch.save(
                        best_model_wts,
                        results_dir + f"/best_model.pt",
                    )
               
            
            print("Epoch {}/{} Loss: {:.2f}, Best Loss: {:.2f}".format(epoch + 1, num_epochs, epoch_loss, best_loss))

            pbar.update(1)

            print(" ")
            print("training loss: ", train_loss_vec)
            print("val loss: ", val_loss_vec)
            print(" ")

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Loss: {:4f}".format(best_loss))

    writer.flush()

    # plot the results to a file
    plot_file = results_dir + "/loss.pdf"
    basic_plot_ts(train_loss_vec, val_loss_vec, plot_file, legend=["Train Loss", "Val Loss"])

    # load best model weights
    model.module.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    train_options = {
        "epochs": 300,
        "learning_rate": 1e-3,
    }

    dataloader_params = {
        "batch_size": 1000,
        "shuffle": True,
        "num_workers": 24,
        "drop_last": False,
        "pin_memory": False,
    }

    print("found device: ", device)

    # condition_list = ['afternoon']

    # experiment_list = [['afternoon'], ['morning'], ['overcast'], ['night'], ['afternoon', 'morning', 'overcast', 'night']]
    experiment_list = [["afternoon"]]

    for condition_list in experiment_list:

        condition_str = "_".join(condition_list)
        print(condition_str)

        Z_DIM = 128
        HISTORY_LEN = 4
        WAYPOINT_LEN = 1
        # where the training results should go
        results_dir = remove_and_create_dir(
            SCRATCH_DIR
            + f"/taxinet_Waypoint{HISTORY_LEN}_Z{Z_DIM}_svae_all_data/"
            # + f"{condition_str}_data_augmentation"
            # + f"{condition_str}"
            # + f"{condition_str}_data_addition"
            # + f"{condition_str}_task_driven"
            + f"{condition_str}_all_data"
            + "/"
        )
        # data_types = [ExprType.orig]
        # data_types = [ExprType.orig, ExprType.morning]
        # data_types = [ExprType.orig, ExprType.morning, ExprType.night, ExprType.overcast]
        data_types = [ExprType.standard]
        # data_types = [ExprType.orig, ExprType.data_augmentation]
        # data_types = [ExprType.orig, ExprType.task_driven]

        # vae_save_path = (
        #     f"/home/pulkit/diffusion-model-based-task-driven-training/xplane_training/trained_models/" f"taxinet_VAE_Z{Z_DIM}_all_data/{condition_str}/best_model_20.pt"
        # )
        vae_save_path = "/home/pulkit/datasets_xplane/".replace("pulkit", DEVICE_NAME)+frac+"/trained_models_svae/taxinet_VAE_Z128_all_data/afternoon/best_model_svae.pt"
        vae_model = SoftIntroVAE(
            cdim=3,
            zdim=Z_DIM,
            channels=[64, 128, 256, 512, 512],
            image_size=IMAGE_HEIGHT,
        ).to(device)
        
        from collections import OrderedDict

        vae_model = torch.nn.DataParallel(vae_model).to(device)
        if vae_save_path is not None:
            chkpt = torch.load(vae_save_path, map_location=torch.device(device))
            vae_model.load_state_dict(chkpt)
            print("VAE Weights Loaded")
            # Freeze VAE
            vae_model = vae_model.module        
        for parameter in vae_model.parameters():
            parameter.requires_grad = False
        vae_model.eval()
        print("Freeze VAE Weights")

        model = PerceptionWaypointModel(
            vae_model,
            input_size=Z_DIM + HISTORY_LEN * 2,
            output_size=WAYPOINT_LEN * 2,
            device="cuda",
            vae_save_path=vae_save_path,
            waypoint_len=WAYPOINT_LEN,
            small=True,
        )
        model = torch.nn.DataParallel(model).to(device)

        # model = torch.nn.DataParallel(model).to(device)
        # waypoint_save_path = os.path.join(
        #     "xplane/scratch/taxinet_Waypoint4_Z128_soft_vae_train/afternoon_task_driven/best_model.pt",
        # )
        # print("Loading from: ", waypoint_save_path)
        # chkpt = torch.load(waypoint_save_path, map_location=torch.device(device))
        # model.load_state_dict(chkpt)
        # print(model)

        # data_dir = f"xplane_data/data_original/{condition_str}/"

        data_dir = "/home/pulkit/datasets_xplane/".replace("pulkit", DEVICE_NAME)+frac+"/xplane_dataset/"
        train_dataset, train_loader = taxinet_prepare_dataloader(
            data_dir,
            condition_str,
            "train",
            dataloader_params,
            HISTORY_LEN,
            None,
            frac,
            data_types,
        )

        val_dataset, val_loader = taxinet_prepare_dataloader(
            data_dir,
            condition_str,
            "val",
            dataloader_params,
            HISTORY_LEN,
            train_dataset.scaler,
            frac,
            data_types,
        )

        # OPTIMIZER
        optimizer = torch.optim.Adam(model.module.parameters(), lr=train_options["learning_rate"], amsgrad=True)

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
            # phases=["val"],
        )

        # save the best model to the directory
        if not os.path.isdir("/home/pulkit/datasets_xplane/".replace("pulkit", DEVICE_NAME)+frac+"/taxinet_wp"):
            os.mkdir("/home/pulkit/datasets_xplane/".replace("pulkit", DEVICE_NAME)+frac+"/taxinet_wp")
        results_dir = "/home/pulkit/datasets_xplane/".replace("pulkit", DEVICE_NAME)+frac+"/taxinet_wp"
        torch.save(model.state_dict(), results_dir + "/best_model_wp.pt")
        break
