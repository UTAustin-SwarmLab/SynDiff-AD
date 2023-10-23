# classify images in the dataset with the testprompts in predict_segment 
# using CLIP.

import sys
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
import clip

current_dir = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_dir)
sys.path.append(parent)

from waymo_open_data_parser.data_loader import WaymoDataset
from waymo_open_data_parser.data_loader import *


# from  import waymo_open_data_parser
# from waymo_open_data_parser.data_loader import dataset
# from waymo_open_data_parser.data_loader import dataloader


device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device)
model.eval()


test_prompts = ["Night, good lighting",
                "Night, bad lighting",
                "Foggy, good lighting",
                "Foggy, bad lighting",
                "Rainy, good lighting",
                "Rainy, bad lighting"]


text_inputs = torch.cat([clip.tokenize(description) for description in test_prompts]).to(device)

config = omegaconf.OmegaConf.load('../Swarm_Lab/segmentation/lang-cond-task-adv-augmentation/lang_cond_task_adv_augmentation/waymo_open_data_parser/config.yaml')
SEGMENTATION = True
IMAGE_META_DATA = False
if config.SAVE_DATA:
    # Append the cwd to the paths in the config file
    # for keys in config.keys():
    #     if 'DIR' in keys:
    #         config[keys] = os.path.join(os.getcwd(), config[keys])

    # Pickle the data
    image_mask_pickler(config, validation=False)
else:
    # Create the dataloader and test the number of images
    dataset = WaymoDataset(config, image_meta_data=IMAGE_META_DATA,
                            segmentation=SEGMENTATION)

    # try except
    try:
        collate_fn = functools.partial(
            waymo_collate_fn, 
            segmentation=SEGMENTATION,
            image_meta_data=IMAGE_META_DATA
        )

        dataloader = DataLoader(
            dataset, 
            batch_size=10,
            shuffle=True, 
            # collate_fn=collate_fn
        )
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)

        print(data[0].shape)
        
    except:
        raise ValueError("The dataloader failed to load the data")


# Define a transformation to resize and format the image
transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to CLIP model's expected input size
        #transforms.ToTensor()  # Convert to a PyTorch tensor
        ])


for camera_images, semantic_mask_rgb, instance_masks, object_masks in dataloader:
    camera_images = camera_images.permute(0, 3, 1, 2)
    camera_images = transform(camera_images)
    images = camera_images.to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_inputs)

    # Calculate similarity scores
    similarity_scores = (image_features @ text_features.T).softmax(dim=1)

    #Print Clip results for each image
    for i, image_file in enumerate(camera_images):
        print(f"Image: {i}")
        for j, prompt in enumerate(test_prompts):
            print(f"Similarity to prompt '{prompt}': {similarity_scores[i, j]:.2f}")



