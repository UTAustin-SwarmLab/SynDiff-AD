import pickle
from torch.utils.data import Dataset, DataLoader
import omegaconf
from tqdm import tqdm
from multiprocess import Pool
from typing import *
import numpy as np
import os
from waymo_open_dataset.utils import camera_segmentation_utils
import torch 
import functools
import tensorflow as tf
from lang_data_synthesis.utils import ADE_20K_PALETTE, COCO_PALETTE
import pandas as pd
from PIL import Image

from lang_data_synthesis.dataset import ExpDataset

class BDD100KDataset(ExpDataset):
    '''
    Loads the dataset from the pickled files
    '''
    CLASSES: str
    PALLETE: str
    FOLDER: str
    # camera_files: List[str]
    # segment_files: List[str]
    # instance_files: List[str]
    num_images: int
    segmentation: bool
     
    def __init__(self, config, 
                 validation=False, 
                 segmentation=True,
                 image_meta_data=False,
                 mixing_ratio = 0.5) -> None:
 

        
        #self.segment_frames = dict()
        self._num_images = 0
        self.validation = validation
        self.segmentation = segmentation
        self.image_meta_data = image_meta_data
        self.bdd_config = config
        self._data_list = []
        
        # Set TRAIN_DIR, ANN_TRAIN_DIR, VAL_DIR , ANN_VAL_DIR
        # self.bdd_config.IMG_DIR = os.path.join(self.bdd_config.TRAIN_DIR, "images/10k/train")
        # self.bdd_config.MASK_DIR = os.path.join(self.bdd_config.VAL_DIR, "images/10k/val")
        # self.bdd_config.ANN_TRAIN_DIR = os.path.join(self.bdd_config.TRAIN_DIR, "labels/sem_seg/masks/train")
        # self.bdd_config.ANN_VAL_DIR = os.path.join(self.bdd_config.VAL_DIR, "labels/sem_seg/masks/val")
                
        # Need to change in accordance with the proposed synthetic dataset structure
        self.metadata_path = None
        if validation:
            if segmentation:
                if os.path.exists(os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv")):
                    self.metadata_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv")
                self.contexts_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                        "filenames_val_seg.txt")
            else:
                if os.path.exists(os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv")):
                    self.metadata_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv")
                self.contexts_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                        "filenames_val_det.txt")
            
            self.bdd_config.IMG_DIR = os.path.join(self.bdd_config.DATASET_DIR, "images/10k/val")
            self.bdd_config.MASK_DIR = os.path.join(self.bdd_config.DATASET_DIR, "labels/sem_seg/masks/val")
        
        elif not validation:
            # Normal data files
            if segmentation:
                if os.path.exists(os.path.join(self.bdd_config.TRAIN_META_PATH,
                                                "metadata_train_seg.csv")):
                    self.metadata_path = os.path.join(self.bdd_config.TRAIN_META_PATH,
                                                "metadata_train_seg.csv")
                self.contexts_path = os.path.join(self.bdd_config.TRAIN_META_PATH,
                                        "filenames_train_seg.txt")
            else:
                if os.path.exists(os.path.join(self.bdd_config.TRAIN_META_PATH,
                                                "metadata_train_det.csv")):
                    self.metadata_path = os.path.join(self.bdd_config.TRAIN_META_PATH,
                                                "metadata_train_det.csv")
                self.contexts_path = os.path.join(self.bdd_config.TRAIN_META_PATH,
                                        "filenames_train_det.txt")
                
            self.bdd_config.IMG_DIR = os.path.join(self.bdd_config.DATASET_DIR, "images/10k/train")
            self.bdd_config.MASK_DIR = os.path.join(self.bdd_config.DATASET_DIR, "labels/sem_seg/masks/train")
            
            # Synthetic data files
            if self.bdd_config.SYNTH_TRAIN_DIR is not None:
                if segmentation:
                    if os.path.exists(os.path.join(self.bdd_config.SYNTH_TRAIN_DIR,
                                                    "metadata_seg.csv")):
                        self.synth_metadata_path = os.path.join(self.bdd_config.SYNTH_TRAIN_DIR,
                                                    "metadata_train_seg.csv")
                    self.synth_contexts_path = os.path.join(self.bdd_config.SYNTH_TRAIN_DIR,
                                            "filenames_seg.txt")
                else:
                    if os.path.exists(os.path.join(self.bdd_config.SYNTH_TRAIN_DIR,
                                                    "metadata_det.csv")):
                        self.synth_metadata_path = os.path.join(self.bdd_config.SYNTH_TRAIN_DIR,
                                                    "metadata_det.csv")
                    self.synth_contexts_path = os.path.join(self.bdd_config.SYNTH_TRAIN_DIR,
                                            "filenames_det.txt")
                                
                self.bdd_config.SYNTH_IMG_DIR = os.path.join(self.bdd_config.SYNTH_TRAIN_DIR,
                                                             "img")
                self.bdd_config.SYNTH_MASK_DIR = os.path.join(self.bdd_config.SYNTH_TRAIN_DIR,
                                                              "mask")
            
        self.mixing_ratio = mixing_ratio
        self.bdd_init(segmentation=segmentation,
                        validation=validation,
                        image_meta_data=image_meta_data)
            
        

    def bdd_init(self, 
                   segmentation: bool = True,
                   validation: bool = False,
                   image_meta_data: bool = False) -> None:

        # Need to change in accordance with the proposed synthetic dataset structure
        self.metadata_path = None
        if validation:
            if segmentation:
                if os.path.exists(os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv")):
                    self.metadata_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv")
                self.contexts_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                        "filenames_val_seg.txt")
            else:
                if os.path.exists(os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_det.csv")):
                    self.metadata_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_det.csv")
                self.contexts_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                        "filenames_val_det.txt")
        elif not validation:
            if segmentation:
                if os.path.exists(os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_train_seg.csv")):
                    self.metadata_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_train_seg.csv")
                self.contexts_path = os.path.join(self.bdd_config.TRAIN_META_PATH,
                                        "filenames_train_seg.txt")
            else:
                if os.path.exists(os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_train_det.csv")):
                    self.metadata_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_train_det.csv")
                self.contexts_path = os.path.join(self.bdd_config.TRAIN_META_PATH,
                                        "filenames_train_det.txt")
        
        self.metadata = None if self.metadata_path is None else pd.read_csv(self.metadata_path)
        self.init_pallete()

        if self.mixing_ratio != 1.0: 
            with open(self.contexts_path, 'r') as f:
                
                for line in f:
                    data_info = {
                        'file_name': line.strip(),
                    }
                    if self.metadata is not None:
                        data_info['condition'] = self.metadata[self.metadata['file_name'] == line.strip()]['condition'].values[0]

                    self._data_list.append(data_info)
                    self._num_images+=1

        if not validation and self.bdd_config.SYNTH_TRAIN_DIR != 'None' and self.bdd_config.SYNTH_TRAIN_DIR is not None:


            if segmentation:
                self.context_path_synth = os.path.join(self.bdd_config.SYNTH_TRAIN_DIR,
                                                "filenames_seg.txt")
                self.metadata_path_synth = os.path.join(self.bdd_config.SYNTH_TRAIN_DIR,
                                                "metadata_seg.csv")
            else:
                self.context_path_synth = os.path.join(self.bdd_config.SYNTH_TRAIN_DIR,
                                                "filenames_det.txt")
                self.metadata_path_synth = os.path.join(self.bdd_config.SYNTH_TRAIN_DIR,
                                                "metadata_det.csv")
            
            added_images = 0
            self.metadata_synth = None if self.metadata_path_synth is None\
                else pd.read_csv(self.metadata_path_synth)
            if self.mixing_ratio != 1.0:
                self.synth_data_length = int(len(self._data_list)/(1 - self.mixing_ratio)) \
                - len(self._data_list)
            else:
                self.synth_data_length = self.metadata_synth.shape[0]

                
            while added_images < self.synth_data_length:
                with open(self.context_path_synth, 'r') as f:
                    
                    for line in f:
                        data_info = {
                            'file_name': line.strip(),
                            'condition': self.metadata_synth[
                                self.metadata_synth['filename'] == line.strip()
                                ]['condition'].values[0],
                            'synthetic': True
                        }
                        self._data_list.append(data_info)
                        self._num_images+=1
                        added_images+=1
            
            
    def init_pallete(self):
        # Define the pallete
        self.CLASSES_TO_PALLETTE = {
            "undefined": [0, 0, 0],
            "road": [128, 64, 128],
            "sidewalk": [244, 35, 232],
            "building": [70, 70, 70],
            "wall": [190, 153, 153],
            "fence": [102, 102, 156],
            "pole": [153, 153, 153],
            "traffic light": [250, 170, 30],
            "traffic sign": [220, 220, 0],
            "vegetation": [107, 142, 35],
            "terrain": [152, 251, 152],
            "sky": [70, 130, 180],
            "person": [220, 20, 60],
            "rider": [255, 0, 0],
            "car": [0, 0, 142],
            "truck": [0, 0, 70],
            "bus": [0, 60, 100],
            "train": [0, 80, 100],
            "motorcycle": [0, 0, 230],
            "bicycle": [119, 11, 32]
        }

        self.CLASSES = list(self.CLASSES_TO_PALLETTE.keys())
        self.PALLETE = list(self.CLASSES_TO_PALLETTE.values())
        self.color_map = np.array(self.PALLETE).astype(np.uint8)
        
                 # Useful for downstream synthesis
        # self.UNMAPPED_CLASSES = ['undefined', 'terrain',
        #                         'trailer','wall', 'fence', 'rider',
        #                          ]
        # Usefu l for semantic mapping
        self.ADE_CLASS_MAPPING = {
                            'sidewalk':'sidewalk',
                            'building':'building',
                            'road':'road',
                            'traffic light': 'traffic light',
                            'vegetation':'tree',
                            'traffic sign':'signboard',
                            'sky':'sky',
                            # 'ground_animal':'animal',
                            # 'pedestrian':'person',
                            'bus': 'bus', 
                            'wall': 'wall',
                            'fence': 'fence',
                            'pole': 'pole',
                            'person': 'person',
                            'car': 'car',
                            'truck': 'truck',
                            'bicycle': 'bicycle',
                            'bus'  : 'bus',
                            }
        self.UNMAPPED_CLASSES_ADE = set(self.CLASSES_TO_PALLETTE.keys())\
            - set(self.ADE_CLASS_MAPPING.keys())
            
        self.COCO_CLASS_MAPPING = { 
                             'road':'road',
                             'bus': 'bus',
                             'train':'train',
                             'truck': 'truck',
                             'bicycle': 'bicycle',
                             'traffic sign': 'stop sign',
                             'traffic light': 'traffic light',
                             'motorcycle':'motorcycle', 
                             'building': 'building-other-merged',
                             'wall': 'wall-other-merged',
                             'fence': 'fence-merged',
                             'person': 'person',
                             'car': 'car',
                             'sky': 'sky-other-merged',
                             'vegetation': 'tree-merged',
                             'sidewalk': 'pavement-merged'
                            }
        
        self.COCO_CLASS_FULL_MAPPING = { 
                        'road':'road',
                        'bus': 'bus',
                        'train':'train',
                        'truck': 'truck',
                        'bicycle': 'bicycle',
                        'traffic sign': 'stop sign',
                        'traffic light': 'traffic light',
                        'motorcycle':'motorcycle', 
                        'building': 'building-other-merged',
                        'wall': 'wall-other-merged',
                        'fence': 'fence-merged',
                        'person': 'person',
                        'car': 'car',
                        'sky': 'sky-other-merged',
                        'vegetation': 'tree-merged',
                        'sidewalk': 'pavement-merged'
                    }
        
        self.UNMAPPED_CLASSES_COCO = set(self.CLASSES_TO_PALLETTE.keys())\
        - set(self.COCO_CLASS_MAPPING.keys())
        
        self.UNMAPPED_CLASSES_COCO_FULL = set(self.CLASSES_TO_PALLETTE.keys())\
        - set(self.COCO_CLASS_FULL_MAPPING.keys())
        
        self.UNMAPPED_CLASSES = self.UNMAPPED_CLASSES_ADE.intersection(
            self.UNMAPPED_CLASSES_COCO)
        
        if self.bdd_config.PALLETE == "bdd100k":
            self.CLASSES_TO_PALLETE_SYNTHETIC = self.CLASSES_TO_PALLETTE
            return
        elif self.bdd_config.PALLETE == "ade20k":
            UNMAPPED_CLASSES = self.UNMAPPED_CLASSES_ADE
            MAPPED_CLASSES = self.ADE_CLASS_MAPPING
            COLORS = ADE_20K_PALETTE
        elif self.bdd_config.PALLETE == "coco":
            UNMAPPED_CLASSES = self.UNMAPPED_CLASSES_COCO
            MAPPED_CLASSES = self.COCO_CLASS_MAPPING
            COLORS = COCO_PALETTE
        elif self.bdd_config.PALLETE == "cocofull":
            UNMAPPED_CLASSES = self.UNMAPPED_CLASSES_COCO_FULL
            MAPPED_CLASSES = self.COCO_CLASS_FULL_MAPPING
            COLORS = COCO_PALETTE
        else:
            UNMAPPED_CLASSES = self.UNMAPPED_CLASSES_ADE
            MAPPED_CLASSES = self.ADE_CLASS_MAPPING
            COLORS = ADE_20K_PALETTE
        
        
        self.CLASSES_TO_PALLETE_SYNTHETIC = {}
        self.UNMAPPED_CLASS_IDX = []
        for j, (key,color)  in enumerate(self.CLASSES_TO_PALLETTE.items()):
            if key in UNMAPPED_CLASSES:
                self.CLASSES_TO_PALLETE_SYNTHETIC[key] = color
                self.UNMAPPED_CLASS_IDX.append(j)
            elif key in MAPPED_CLASSES.keys():
                self.CLASSES_TO_PALLETE_SYNTHETIC[key] = COLORS[MAPPED_CLASSES[key]]
        
        self.UNMAPPED_CLASS_IDX = np.array(self.UNMAPPED_CLASS_IDX)
        self.color_map_synth = np.array(
            list(self.CLASSES_TO_PALLETE_SYNTHETIC.values()))\
                .astype(np.uint8)  
    
    def _load_item(self, data:dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
       
        if 'condition' in data.keys():
            condition = data['condition']
        
        with torch.device('cpu'):
            file_name = data['file_name']
            if 'synthetic' in data.keys():
                img_path = os.path.join(self.bdd_config.SYNTH_IMG_DIR,
                                        file_name + '.png')
                ann_path = os.path.join(self.bdd_config.SYNTH_MASK_DIR,
                                        file_name + '.png')
            else:
                img_path = os.path.join(self.bdd_config.IMG_DIR,
                                        file_name + '.jpg')
                ann_path = os.path.join(self.bdd_config.MASK_DIR,
                                        file_name + '.png')
            
            camera_images = np.array(Image.open(img_path)).astype(np.uint8)
            if self.segmentation:
                object_masks = np.array(Image.open(ann_path)).astype(np.uint8)
                if 'synthetic' not in data.keys():
                    object_masks = (object_masks + 1)%256
                instance_masks = object_masks.copy()
                semantic_mask_rgb = self.get_semantic_mask(object_masks)
            else:
                raise NotImplementedError
            
            img_data = data
    

            if self.segmentation:
                if self.image_meta_data:
                    return camera_images, semantic_mask_rgb, instance_masks, object_masks, img_data
                else:
                    return camera_images, semantic_mask_rgb, instance_masks, object_masks
            else:
                raise NotImplementedError
                if self.image_meta_data:
                    return camera_images, box_classes, bounding_boxes, img_data
                else:
                    return camera_images, box_classes, bounding_boxes
    
    def __getitem__(
            self, 
            index:int
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        '''
        Returns an item from the dataset referenced by index

        Args:
            index: The index of the item to return
        Returns:
            camera_images: The camera images
            semantic_mask_rgb: The semantic mask in rgb format
            instance_masks: The instance masks
            object_masks: The object masks
            img_data (dict): The image data
        '''
        
        # object_masks = self.get_object_class(semantic_masks)

        if index >= self._num_images:
            raise IndexError("Index out of range")

        data = self._data_list[index]
        
        return self._load_item(data)
        
    