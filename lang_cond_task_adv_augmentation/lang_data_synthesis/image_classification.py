# classify images in the dataset with the testprompts in predict_segment 
# using CLIP.

import sys
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
import open_clip

current_dir = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_dir)
sys.path.append(parent)
from copy import copy
from waymo_open_data import WaymoDataset
from lang_data_synthesis.dataset import collate_fn as collator
import omegaconf
import functools
import torch.utils.data.dataloader as DataLoader
from pandas import DataFrame
import pandas as pd
from tqdm import tqdm
import numpy as np
import tensorflow as tf
# from  import waymo_open_data_parser
# from waymo_open_data_parser.data_loader import dataset
# from waymo_open_data_parser.data_loader import dataloader
import open_clip
from lang_data_synthesis.utils import write_to_csv_from_dict
from copy import deepcopy
from argparse import ArgumentParser

from bdd100k.data_loader import BDD100KDataset

class CLIPClassifier:
    def __init__(self, 
                 config: omegaconf,
                 dataset: WaymoDataset):
        self.config = config
        self.dataset: WaymoDataset = dataset
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, _ , transform = open_clip.create_model_and_transforms(
            'hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
            device=self.device
        ) 
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(
            'hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'
        )
        
        self.prompts = deepcopy(self.config.ROBUSTIFICATION.test_conditions)
        for j, description in enumerate(self.config.ROBUSTIFICATION.test_conditions):
            for i, conditions in enumerate(description):
                self.prompts[j][i] = self.config.ROBUSTIFICATION.prompt_add[j].format(conditions)
        self.text_inputs = torch.cat([self.tokenizer(description) for description \
            in self.prompts]).to(self.device)
        
        self.text_features = self.model.encode_text(self.text_inputs)
        # conditions_list = sum(self.config.ROBUSTIFICATION.test_prompts_conditions,[])
        # self.text_inputs_conditions = torch.cat([clip.tokenize(description) for description \
        #       in conditions_list]).to(self.device)
        #self.text_features_conditions = self.model.encode_text(self.text_inputs_conditions)
        self.transform = transforms.Compose([
        transform.transforms[0],
        transform.transforms[1],
        transform.transforms[-1]# Convert to a PyTorch tensor
        ])
        
        collate_fn = functools.partial(
                collator, 
                segmentation=self.dataset.segmentation,
                image_meta_data=self.dataset.image_meta_data
            )
        self.dataloader = DataLoader.DataLoader(self.dataset,
                                            batch_size=self.config.ROBUSTIFICATION.batch_size,
                                            shuffle=False,
                                            collate_fn=collate_fn,
                                            num_workers=self.config.ROBUSTIFICATION.batch_size)
        
        if isinstance(dataset, WaymoDataset):
            data_dict = {
                "context_name":"content_name",
                "context_frame":"context_frame",
                "camera_id":"camera_id",
                # "image_index":"image_index",
                "condition":"condition"
            }
        elif isinstance(dataset, BDD100KDataset):
            data_dict = {
                "file_name":"file_name",
                "condition":"condition"
            }
            
        write_to_csv_from_dict(
                    dict_data=data_dict , 
                    csv_file_path= self.config.FILENAME,
                    file_name=""
                )
    
    def classify_image(
        self,
        image: np.ndarray
    ):
        camera_images = image.transpose(0, 3, 1, 2)
        camera_images = self.transform(torch.tensor(camera_images).to(torch.float32)/255)
        images = camera_images.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(images)

    # Calculate similarity scores
        #similarity_scores = (image_features @ self.text_features.T).softmax(dim=1)
        
        similarity_scores = (image_features @ self.text_features.T)
        
        condition_batch = ["" for j in range(self.config.ROBUSTIFICATION.batch_size)]
        for j in range(self.config.ROBUSTIFICATION.batch_size):
            condition = ""
            cum_sum = 0
            for classes in self.config.ROBUSTIFICATION.test_conditions:
                score = similarity_scores[[j],cum_sum:cum_sum+len(classes)].softmax(dim=1)
                cum_sum += len(classes)
                
                condition += (classes[torch.argmax(score, dim=1).item()] + ", ")
            condition_batch[j] = condition[:-2]
        return condition_batch

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
        # Load from dataloader
        image_indices = 0
        for j, outputs in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            camera_images = outputs[0]
            condition = []
            time = None
            if self.dataset.image_meta_data:
                image_data = outputs[4]
                condition = None
                # Only for waymo
                if 'condition' in image_data[0].keys() and 'WAYMO' in config.IMAGE.keys():
                    condition = [img_data['condition'] for img_data in image_data]
                    time = [cond.split(",")[1] for cond in condition]

            condition_class = self.classify_image(camera_images.numpy())  
            condition = [] 
            if time is not None and 'WAYMO' in config.IMAGE.keys():
                for t, cond in zip(time, condition_class):
                    cond = cond + "," + t
                    condition.append(cond)
            else:
                condition = condition_class
            
            # Add to df
            for idx in range(image_indices,image_indices+len(image_data)):
                # image_data[idx - image_indices]['index'] = idx
                image_data[idx - image_indices]['condition'] = condition[idx - image_indices]
                write_to_csv_from_dict(
                    dict_data = image_data[idx - image_indices], 
                    csv_file_path= self.config.FILENAME,
                    file_name=""
                )
                
            image_indices += len(image_data)
        return None


def parse_args():
    parser = ArgumentParser(description='Image Classification with CLIP')

    parser.add_argument(
        '--validation',
        action='store_true',
        default=False,
        help='Use validation dataset')
    parser.add_argument(
        '--segmentation',
        action='store_true',
        default=False,
        help='Enable it for the segmentation dataset')
    parser.add_argument(
        '--img_meta_data',
        action = 'store_true',
        default = False,
        help = 'Enable it for the image meta data dataset')
        
    parser.add_argument(
        '--experiment',
        choices=['waymo', 'bdd', 'cliport'],
        default='none',
        help='Which experiment config to generate data for')

    return parser.parse_args()

if __name__=="__main__":
    

    args = parse_args()
    
    SEGMENTATION = args.segmentation
    IMAGE_META_DATA = args.img_meta_data
    VALIDATION = args.validation
    
    tf.config.set_visible_devices([], 'GPU')
    
    config_file_name = 'lang_data_synthesis/{}_config.yaml'.format(args.experiment)
    config = omegaconf.OmegaConf.load(config_file_name)
    if args.validation:
        FILENAME = config.ROBUSTIFICATION.val_file_path
    else:
        FILENAME = config.ROBUSTIFICATION.train_file_path
        
    if os.path.exists(FILENAME) and not config.ROBUSTIFICATION.classify:
        # Compute and show the number of images for each condition
        df = pd.read_csv(FILENAME)
        print(df.columns)
        print(df.groupby(['condition']).size()/len(df)*100)
    else:
        if args.experiment == 'waymo':
            dataset = WaymoDataset(config.IMAGE.WAYMO, 
                                image_meta_data=IMAGE_META_DATA,
                                segmentation=SEGMENTATION,
                                validation=VALIDATION)
        elif args.experiment == 'bdd':
            dataset = BDD100KDataset(config.IMAGE.BDD, 
                                image_meta_data=IMAGE_META_DATA,
                                segmentation=SEGMENTATION,
                                validation=VALIDATION)
        elif args.experiment == 'plan':
            raise NotImplementedError
        elif args.experiment == 'cliport':
            raise NotImplementedError
        else:
            raise ValueError("Experiment not supported")

        config.FILENAME = FILENAME
        classifier = CLIPClassifier(config, 
                                    dataset)
        df = classifier.classify()
        # print(df.columns)
        # # Save the dataframe
        # df.to_csv(FILENAME, index=False)


