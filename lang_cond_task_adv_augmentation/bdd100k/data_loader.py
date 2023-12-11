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
class BDDDataset(Dataset):
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
                 mixing_ratio = 1.0) -> None:
 

        
        #self.segment_frames = dict()
        self._num_images = 0
        self.ds_config = config
        self.validation = validation
        self.segmentation = segmentation
        self.image_meta_data = image_meta_data
        self.bdd_config = config
        self.data_list = []
        
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
        elif not validation:
            # Normal data files
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

        self.mixing_ratio = mixing_ratio
        self.bdd_init(segmentation=segmentation,
                        validation=validation,
                        image_meta_data=image_meta_data)
            
        

    def bdd_init(self, 
                   segmentation: bool = True,
                   validation: bool = False,
                   image_meta_data: bool = False) -> None:
        
        self.synth_data_length = int(len(self.data_list)/(1 - self.mixing_ratio)) \
            - len(self.data_list)
        
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
                                                "metadata_val_seg.csv")):
                    self.metadata_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv")
                self.contexts_path = os.path.join(self.bdd_config.TRAIN_META_PATH,
                                        "filenames_train_seg.txt")
            else:
                if os.path.exists(os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_det.csv")):
                    self.metadata_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_det.csv")
                self.contexts_path = os.path.join(self.bdd_config.TRAIN_META_PATH,
                                        "filenames_train_det.txt")
        
        self.metadata = None if self.metadata_path is None else pd.read_csv(self.metadata_path)
        self.init_pallete()

        with open(self.contexts_path, 'r') as f:
            
            for line in f:
                data_info = {
                    'file_name': line.strip(),
                }
                if self.metadata is not None:
                    data_info['condition'] = self.metadata[self.metadata['file_name'] == line.strip()]['condition'].values[0]

                self.data_list.append(data_info)
                self._num_images+=1
                added_images+=1

        if not validation and self.bdd_config.SYNTH_TRAIN_DIR is not None:
            self.synth_data_length = int(len(self.data_list)/(1 - self.mixing_ratio)) \
                - len(self.data_list)

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
            self.metadata_synth = None if self.metadata_path_synth is None else pd.read_csv(self.metadata_path_synth)
            while added_images < self.synth_data_length:
                with open(self.contexts_path_synth, 'r') as f:
                    
                    for line in f:
                        data_info = {
                            'file_name': line.strip(),
                            
                        }
                        self.data_list.append(data_info)
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
                             'sidewalk': 'pavement-other-merged'
                            }
        
        self.UNMAPPED_CLASSES_COCO = set(self.CLASSES_TO_PALLETTE.keys())
        - set(self.COCO_CLASSES.keys())
        
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
    
    def get_semantic_mask(self, object_mask: np.ndarray) -> np.ndarray:
        '''
        Returns the semantic mask from the object mask

        Args:
            object_mask: The object mask to extract the semantic mask from
        
        Returns:
            semantic_mask: The semantic mask of the object mask
        '''
        semantic_mask = self.color_map[object_mask.squeeze()]
        return semantic_mask

    def get_mapped_semantic_mask(self, 
                                 object_mask: np.ndarray) -> np.ndarray:
        '''
        Returns the semantic mask from the object mask mapped to the ADE20K classes
        or COCO semantic classes

        Args:
            object_mask: The object mask to extract the semantic mask from
        
        Returns:
            semantic_mask: The semantic mask of the object mask
        '''
        
        # convert all the ade20k objects in the color mask to
        semantic_mask = self.color_map_synth[object_mask.squeeze()]
        return semantic_mask
        
    def get_unmapped_mask(self,
                          object_mask: np.ndarray) -> np.ndarray:
        '''
        Returns a boolean mask whether the object mask category is mapped or not
        '''
        mask = np.zeros(object_mask.shape, dtype=np.bool)
        for idx in self.UNMAPPED_CLASS_IDX:
            mask = np.logical_or(mask, object_mask == idx)
        return mask
    
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

        context_frame = self._data
        
        with tf.device('cpu'):
            if self.segmentation:
                data = load_data_set_parquet(
                    config=self.ds_config, 
                    context_name=context_name, 
                    validation=self.validation,
                    context_frames=[context_frame],
                    return_weather_cond=self.image_meta_data
                )
                if self.image_meta_data:
                    frames_with_seg, camera_images, weather_list = data
                else:
                    frames_with_seg, camera_images = data

                semantic_labels_multiframe, \
                instance_labels_multiframe, \
                panoptic_labels = read_semantic_labels(
                    self.ds_config,
                    frames_with_seg
                )
                
                data = read_camera_images(
                    self.ds_config,
                    camera_images,
                    return_weather_cond=self.image_meta_data,
                    weather_conditions=weather_list
                )

                if self.image_meta_data:
                    camera_images_frame, weather_labels_frame, lighting_conditions_frame = data
                    weather = weather_labels_frame[0][camera_id]
                    lighting_conditions = lighting_conditions_frame[0][camera_id]
                    condition = weather + ', ' + lighting_conditions
                else:
                    camera_images_frame = data    
                # All semantic labels are in the form of object indices defined by the PALLETE
                camera_images = camera_images_frame[0][camera_id]
                object_masks = semantic_labels_multiframe[0][camera_id].astype(np.int64)
                instance_masks = instance_labels_multiframe[0][camera_id].astype(np.int64)

                semantic_mask_rgb = self.get_semantic_mask(object_masks)
                panoptic_mask_rgb = camera_segmentation_utils.panoptic_label_to_rgb(object_masks,
                                                                                instance_masks)
            else:
                boxes, camera_images = load_data_set_parquet(
                    config=self.ds_config, 
                    context_name=context_name, 
                    validation=self.validation,
                    context_frames=[context_frame],
                    segmentation=False,
                    return_weather_cond=self.image_meta_data
                )
                if self.image_meta_data:
                    frames_with_seg, camera_images, weather_list = data
                else:
                    frames_with_seg, camera_images = data

                box_classes, bounding_boxes = read_box_labels(
                    self.ds_config,
                    boxes
                )
                
                camera_images_frame = read_camera_images(
                    self.ds_config,
                    camera_images,
                    return_weather_cond=self.image_meta_data,
                    weather_conditions=weather_list
                )

                if self.image_meta_data:
                    camera_images_frame, weather_labels_frame, lighting_conditions_frame = data
                    weather = weather_labels_frame[0][camera_id]
                    lighting_conditions = lighting_conditions_frame[0][camera_id]
                    condition = weather + ', ' + lighting_conditions
                else:
                    camera_images_frame = data  

                camera_images = camera_images_frame[0][camera_id]
                box_classes = box_classes[camera_id][0]
                bounding_boxes = bounding_boxes[camera_id][0]
        
        if self.image_meta_data:
            img_data = {
                'context_name': context_name,
                'context_frame': context_frame,
                'camera_id': camera_id,
                'condition': condition
            }

        if self.segmentation:
            if self.image_meta_data:
                return camera_images, semantic_mask_rgb, instance_masks, object_masks, img_data
            else:
                return camera_images, semantic_mask_rgb, instance_masks, object_masks
        else:
            if self.image_meta_data:
                return camera_images, box_classes, bounding_boxes, img_data
            else:
                return camera_images, box_classes, bounding_boxes
    
    def __len__(self) -> int:
        # return max(len(self.camera_files), 
        #             len(self.segment_files), 
        #             len(self.instance_files))
        return self._num_images

    @property
    def METADATA(self):
        return self._data_list
    
    @staticmethod
    def get_text_description(object_mask: np.ndarray, CLASSES) -> str:
        '''
        Returns the text description of the object mask

        Args:
            object_mask: The object mask to extract the text description from
        
        Returns:
            text_description: The text description of the object mask
        '''
        
        object_set = set(object_mask.flatten().tolist())
        text_description = ''
        for object in object_set:
            text_description += CLASSES[object] + ', '
        return text_description[:-1]