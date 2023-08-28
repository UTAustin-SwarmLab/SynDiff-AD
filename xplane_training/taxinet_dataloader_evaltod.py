import os
import torch
import numpy as np
import h5py
from PIL import Image
import pickle
import sys
from typing import List
from IPython import embed
from sklearn.preprocessing import StandardScaler
from enum import Enum, auto
import random


from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from xplane_training.utils import SEED, ENDC, OKBLUE

# make sure this is a system variable in your bashrc
# NASA_ULI_ROOT_DIR = os.environ["NASA_ULI_ROOT_DIR"]

# DATA_DIR = os.environ["NASA_DATA_DIR"]

# where intermediate results are saved
# never save this to the main git repo
# SCRATCH_DIR = NASA_ULI_ROOT_DIR + "/scratch/"
torch.manual_seed(SEED)

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
NUM_ADVER_EXAMPLES = 15004
NUM_TRAINING_OOD_SAMPLES = 4000
NUM_VAL_OOD_SAMPLES = 1000

condition_dict = {"morning": 0, "afternoon": 1, "night": 2, "adver": 3}


class ExprType(Enum):
    orig = auto()
    morning = auto()
    night = auto()
    overcast = auto()
    task_agnostic = auto()
    task_driven = auto()
    data_augmentation = auto()
    standard  = auto()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.255]

transform_normalize = transforms.Compose(
    [
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

transform_denormalize = transforms.Compose(
    [
        transforms.Normalize(
            mean=[-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]],
            std=[1 / std[0], 1 / std[1], 1 / std[2]],
        ),
    ]
)


def get_data_from_dir(
    tod: str,
    dataloader_dir: str,
    history_len: int,
    train_test_split: str,
    scaler: StandardScaler,
    
):
    image_dir = "/home/pulkit/xplane/"+tod+"/"+tod+"_validation/"
    image_list = [x for x in os.listdir(image_dir) if x.endswith(".png")]
    print("Length of image list: ", len(image_list))
    dataloader_dir = "/home/pulkit/diffusion-model-based-task-driven-training/xplane_training/"
    with open(os.path.join(dataloader_dir, tod+"_valwaypoint_data.pickle"), "rb") as handle:
        waypoint_dict = pickle.load(handle)
    # print(waypoint_dict)
    all_images = list(set(image_list).intersection(set(list(waypoint_dict.keys()))))
    print("LENGTH OF WP DICT", len(waypoint_dict))

    waypoint_len = history_len + 1
    flatten_data = np.swapaxes(np.array(list(waypoint_dict.values())), 1, 2)
    assert (
        flatten_data.shape[1] == waypoint_len
    ), f"Waypoint {flatten_data.shape} Len Mismatch with expected {waypoint_len}"
    flatten_data = flatten_data.reshape(flatten_data.shape[0] * flatten_data.shape[1], flatten_data.shape[2])
  
    if train_test_split == "train":
        train_scaler = StandardScaler()
        print("Fitting scaler")
        norm_flatten_data = train_scaler.fit_transform(flatten_data)

    elif train_test_split == "validation" or train_test_split == "test":
        norm_flatten_data = scaler.transform(flatten_data)
        train_scaler = scaler

    for idx, k in enumerate(waypoint_dict.keys()):
        waypoint_dict[k] = norm_flatten_data[idx * waypoint_len : (idx + 1) * waypoint_len]
        
    # print(all_images)
    return all_images, waypoint_dict, train_scaler


class TaxiNetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tod,
        dataloader_dir,
        train_test_split: str,
        history_len: int,
        scaler: StandardScaler,
        exper_types: List[ExprType] = [ExprType.orig],
    ):
        # print(history_len)
        self.all_images = []
        self.waypoint_dict = {}
        # image transforms
        self.tfms = transform_normalize
        self.history_len = history_len
        self.train_data_dir = dataloader_dir
        self.all_data_paths = [dataloader_dir]

        # for exper_type in exper_types:
        #     if exper_type == ExprType.standard:
                # update the waypoint directory and do stuff as in ExprType.orig
        print("doing")
        images, waypoint_dict, self.scaler = get_data_from_dir(tod, dataloader_dir, history_len, train_test_split, scaler)
        print(self.scaler)
        self.all_images = self.all_images + images
        print("Length of all images: ", len(self.all_images))
        self.waypoint_dict.update(waypoint_dict)
        self.all_images.sort()

    def __len__(self):
        # number of images in dataset
        num_images = len(self.all_images)
        return num_images

    def __getitem__(self, index):
        "Generates one sample of data"

        image_name = self.all_images[index]

        condition_label = None
        for cond in condition_dict.keys():
            if cond in image_name:
                condition_label = condition_dict[cond]

        assert condition_label is not None

        def get_image_from_path(fname):
            image = Image.open(fname).convert("RGB")
            tensor_image_example = self.tfms(image)
            return tensor_image_example

        # open images and apply transforms
        tensor_image_example = None
        if "adver" not in image_name:
            for path in self.all_data_paths:
                if os.path.exists(path + "/" + str(image_name)):
                    tensor_image_example = get_image_from_path(os.path.join(path + "/" + str(image_name)))
                    break
            if tensor_image_example is None:
                print("Image not found")
                raise AssertionError

        elif "adver" in image_name:
            image = torch.from_numpy(self.adver_images[image_name]).type(torch.FloatTensor)
            tensor_image_example = self.tfms.transforms[2](image)
        else:

            print("Image not found")
            raise AssertionError

        # concatenate all image tensors
        target_tensor = torch.Tensor(self.waypoint_dict[image_name]).type(torch.FloatTensor)
        history = target_tensor[: self.history_len]
        target_point = target_tensor[-1]

        return tensor_image_example, history.flatten(), target_point, condition_label

    def inverse_transform_torch(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[1] == 2

        mean = self.scaler.mean_
        var = self.scaler.scale_

        output = torch.zeros_like(input)
        output[:, 0] = input[:, 0] * var[0] + mean[0]
        output[:, 1] = input[:, 1] * var[1] + mean[1]

        return output

###################################################################################################################################################
###################################################################################################################################################
def taxinet_prepare_dataloader(
    tod,
    data_dir: str,
    condition_str: str,
    train_test_split: str,
    params: dict,
    history_len: int,
    train_scaler: StandardScaler,
    exper_types: List[ExprType] = [ExprType.orig],
):
    # data_dir = os.path.join(data_dir + train_test_split)
    exper_types = [ExprType.standard]
    # print(data_dir)
    # assert isinstance(exper_types, list)
    data_dir = "/home/pulkit/xplane/" + tod + "/" + tod + "_validation/"
    print(history_len)
    taxinet_dataset = TaxiNetDataset(tod, data_dir, train_test_split, history_len, train_scaler, exper_types)
    
    print(" ")
    print(
        "condition: ",
        condition_str,
        ", train_test_split: ",
        train_test_split,
        ", Expr Type: ",
        [exper_type.name for exper_type in exper_types],
    )
    print(f"Dataset Size: {len(taxinet_dataset)}")
    print("X shape: ", taxinet_dataset[0][0].shape)
    print("y shape: ", taxinet_dataset[0][1].shape)
    print(" ")

    taxinet_dataloader = DataLoader(taxinet_dataset, **params)
    return taxinet_dataset, taxinet_dataloader


