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
from copy import copy
from waymo_open_data import WaymoDataset, waymo_collate_fn
import omegaconf
import functools
import torch.utils.data.dataloader as DataLoader
from pandas import DataFrame
import pandas as pd
from tqdm import tqdm
import numpy as np
# from  import waymo_open_data_parser
# from waymo_open_data_parser.data_loader import dataset
# from waymo_open_data_parser.data_loader import dataloader


class CLIPClassifier:
    def __init__(self, 
                 config: omegaconf,
                 dataset: WaymoDataset):
        self.config = config
        self.dataset: WaymoDataset = dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.transform = clip.load("ViT-L/14@336px", self.device)    
        self.model.eval()
        self.text_inputs = torch.cat([clip.tokenize(description) for description \
            in self.config.ROBUSTIFICATION.test_prompts]).to(self.device)
        
        # conditions_list = sum(self.config.ROBUSTIFICATION.test_prompts_conditions,[])
        # self.text_inputs_conditions = torch.cat([clip.tokenize(description) for description \
        #       in conditions_list]).to(self.device)
        
        #self.text_inputs = self.config.TEST_PROMPTS.test_prompts
        self.text_features = self.model.encode_text(self.text_inputs)
        # self.text_features_conditions = self.model.encode_text(self.text_inputs_conditions)
        self.transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.CenterCrop((336,336)),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))# Convert to a PyTorch tensor
        ])
    
    def classify_image(
        self,
        image: np.ndarray
    ):
        camera_images = image.transpose(0, 3, 1, 2)
        camera_images = self.transform(torch.tensor(camera_images).to(torch.float32)/256)
        images = camera_images.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(images)

    # Calculate similarity scores
        #similarity_scores = (image_features @ self.text_features.T).softmax(dim=1)
        
        similarity_scores = (image_features @ self.text_features.T)
        
        cum_sum = 0
        condition = ""
        for classes in self.config.ROBUSTIFICATION.test_prompts:
            score = similarity_scores[:,cum_sum:cum_sum+len(classes)].softmax(dim=1)
            cum_sum += len(classes)
            
            condition += (classes[torch.argmax(score, dim=1).item()] + ",")
            
        return copy(condition[:-1])

    def classify(
        self
    ):

        # Create a dataframe where we have 4 columns
        # 1. Image index
        # 2. Image camera id
        # 
        # waymo_collate_fn = functools.partial(
        #     waymo_collate_fn,
        #     image_meta_data=self.dataset.image_meta_data,
        #     segmentation=self.dataset.segmentation
        # )
        
        # dataloader = DataLoader.DataLoader(self.dataset,
        #                                     batch_size=len(self.dataset),
        #                                     shuffle=False,
        #                                     collate_fn=waymo_collate_fn,
        #                                     num_workers=0)
        df = DataFrame(columns=['image_index', 
                                'context_name',
                                'context_frame',
                                'camera_id',
                                'condition'])
        
        for j in tqdm(range(len(self.dataset))):
        # for j in tqdm(100):
            outputs = self.dataset[j]
            camera_images = outputs[0]
            semantic_mask_rgb = outputs[1]
            instance_masks = outputs[2]
            object_masks = outputs[3]
            if self.dataset.image_meta_data:
                image_data = outputs[4]

            # condition = self.classify_image(torch.tensor(camera_images,
            #                                           dtype=torch.float32
            #                                         ).unsqueeze(0))  
            condition = self.classify_image(np.expand_dims(camera_images, axis=0))        
            # Add to df

            df = df.append({'image_index': j,
                            'context_name': image_data['context_name'],
                            'context_frame': image_data['context_frame'],
                            'camera_id': image_data['camera_id'],
                            'condition': condition}, ignore_index=True)
 
        return df
    
if __name__=="__main__":
    config = omegaconf.OmegaConf.load('lang_data_synthesis/config.yaml')
    SEGMENTATION = True
    IMAGE_META_DATA = True
    VALIDATION = True
    
    if VALIDATION:
        FILENAME = "waymo_open_data/waymo_clip_classification_small_val.csv"
    else:
        FILENAME = "waymo_open_data/waymo_clip_classification_small.csv"
    if os.path.exists(FILENAME):
        # Compute and show the number of images for each condition
        df = pd.read_csv(FILENAME)
        print(df.columns)
        print(df.groupby(['condition']).size())
    else:
        dataset = WaymoDataset(config.IMAGE.WAYMO, 
                                image_meta_data=IMAGE_META_DATA,
                                segmentation=SEGMENTATION,
                                validation=VALIDATION)
        classifier = CLIPClassifier(config, dataset)
        df = classifier.classify()
        print(df.columns)
        
        # Save the dataframe
        df.to_csv(FILENAME, index=False)


