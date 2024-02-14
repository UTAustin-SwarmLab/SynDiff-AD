import pickle
from typing import Tuple
from torch._tensor import Tensor
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

from lang_data_synthesis.dataset import ExpDataset, collate_fn
import json
class CARLADataset(ExpDataset):
    '''
    Loads the dataset from the carla data folders
    '''
    CLASSES: str
   
    """
    Dataset class for images, point clouds and vehicle control in CARLA.
    """
    def __init__(self,
                 config,
                 validation=False, 
                 segmentation=True,
                 image_meta_data=False
                 ):
        
        #  Preset config for the dataset from carla training
        carla_config = config.IMAGE.CARLA
        self.seq_len = carla_config.seq_len
        self.pred_len = carla_config.pred_len
        self.tot_len = carla_config.tot_len
        self.points_per_class = carla_config.points_per_class
        self.scale = carla_config.scale
        self.crop = carla_config.crop
        self.scale_topdown = carla_config.scale_topdown
        self.crop_topdown = carla_config.crop_topdown
        self.num_class = carla_config.num_class
        self.converter = carla_config.converter
        self.t_height = carla_config.t_height
        self.axis = carla_config.axis
        self.resolution = carla_config.resolution
        self.offset = carla_config.offset
        
        self.image_meta_data = image_meta_data
        self.segmentation = segmentation

        # initialize preload lists
        self._data_list = []
        # Data info for each info we have the source route, 
        # the image file name and the semantic annotation file name
        root = []
        if validation:
            for town in carla_config.val_towns:
                root.append(os.path.join(carla_config.root_dir, town))
        else:
            for town in carla_config.train_towns:
                root.append(os.path.join(carla_config.root_dir, town))
                
        for sub_root in root:
            
            if not os.path.exists(sub_root):
                continue
            
            #preload_file = os.path.join(sub_root, 'pl_'+str(carla_config.seq_len)+'_'+str(carla_config.pred_len)+'.npy')
            if 'synth' in sub_root:
                synthetic = True
                
            else:
                synthetic = False
            
            root_files = os.listdir(sub_root)
            routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
            for route in routes:
                route_dir = os.path.join(sub_root, route)
                print(route_dir)
                
                if not os.path.exists(route_dir+"/rgb_front/"):
                    continue
                
                num_seq = len(os.listdir(route_dir+"/rgb_front/"))
                for seq in range(num_seq):
                    
                    filename = f"{str(seq).zfill(4)}.png"
                    if synthetic:
                        # if synthetic, the masks for the images will be found in the 
                        # default expert data folder
                        # route will be named synth___route___1, synth___route___2 etc
                        mask_home_root = root.split('___')[1]
                        mask_route_dir = route_dir.replace('synth', '')
                        mask_route_dir = mask_route_dir.replace(route, mask_home_root)
                    else:
                        mask_route_dir = route_dir  
                    
                    # Open measurements.json to load weather condition
                    with open(route_dir + f"/measurements/{str(seq).zfill(4)}.json", "r") as read_file:
                        data = json.load(read_file)
                        conditions = data['weather']
                    data_dict = {
                        'route': route,
                        'file_name':route_dir+"/rgb_front/"+filename,
                        'mask_path': mask_route_dir+"/seg_front/"+filename,
                        'synthetic': synthetic,
                        'condition': conditions
                    }
                    
                    self._data_list.append(data_dict)
                    data_dict = {
                        'route': route,
                        'file_name':route_dir+"/rgb_left/"+filename,
                        'mask_path':  mask_route_dir+"/seg_left/"+filename,
                        'synthetic': synthetic,
                        'condition': conditions
                    }
                    
                    self._data_list.append(data_dict)
                    
                    data_dict = {
                        'route': route,
                        'file_name':route_dir+"/rgb_right/"+filename,
                        'mask_path': mask_route_dir+"/seg_right/"+filename,
                        'synthetic': synthetic,
                        'condition': conditions
                    }
                    self._data_list.append(data_dict)
        self._num_images = len(self._data_list)
        self.carla_init()
        
    @property
    def data_keys(self):
        return self._data_list[0].keys()
            
    def carla_init(self):
        
        # Initialise the color mappings in the carla dataset
        # We refer to sensitive classes as unmapped classes and during synthesis these classes
        # will not be synthetic, we will superpose them on the synthetic image from the original class
        label_data = [
            # {"Class": "Buffer", "Color": (0, 0, 0)},
            {"Class": "Unlabeled", "Color": (0, 0, 0)},
            {"Class": "Building", "Color": (70, 70, 70)},
            {"Class": "Fence", "Color": (100, 40, 40)},
            {"Class": "Other", "Color": (55, 90, 80)},
            {"Class": "Pedestrian", "Color": (220, 20, 60)},
            {"Class": "Pole", "Color": (153, 153, 153)},
            {"Class": "RoadLine", "Color": (157, 234, 50)},
            {"Class": "Road", "Color": (128, 64, 128)},
            {"Class": "SideWalk", "Color": (244, 35, 232)},
            {"Class": "Vegetation", "Color": (107, 142, 35)},
            {"Class": "Vehicles", "Color": (0, 0, 142)},
            {"Class": "Wall", "Color": (102, 102, 156)},
            {"Class": "TrafficSign", "Color": (220, 220, 0)},
            {"Class": "Sky", "Color": (70, 130, 180)},
            {"Class": "Ground", "Color": (81, 0, 81)},
            {"Class": "Bridge", "Color": (150, 100, 100)},
            {"Class": "RailTrack", "Color": (230, 150, 140)},
            {"Class": "GuardRail", "Color": (180, 165, 180)},
            {"Class": "TrafficLight", "Color": (250, 170, 30)},
            {"Class": "Static", "Color": (110, 190, 160)},
            {"Class": "Dynamic", "Color": (170, 120, 50)},
            {"Class": "Water", "Color": (45, 60, 150)},
            {"Class": "Terrain", "Color": (145, 170, 100)},
            {"Class": "Red Light", "Color": (255, 0, 0)},
            {"Class": "Yellow Light", "Color": (255, 0, 0)},
            {"Class": "Green Light", "Color": (0, 255, 0)},
            {"Class": "Stop Sign", "Color": (220, 220, 0)}   
        ]        
        self.CLASSES_TO_PALLETE = {data["Class"]: data["Color"] for data in label_data}
        self.CLASSES = list(self.CLASSES_TO_PALLETE.keys())
        self.PALLETE = list(self.CLASSES_TO_PALLETE.values())
        self.color_map = np.array(self.PALLETE).astype(np.uint8)
        
        # We physically list out the classes that we dont want to synthetically map
        self.UNMAPPED_CLASSES=[
            'Unlabeled','TrafficSign','TrafficLight','Stop Sign',
            'Red Light','Green Light','Yellow Light', 'RoadLine'
        ]
        
        UNMAPPED_CLASSES = self.UNMAPPED_CLASSES
    
        self.CLASSES_TO_PALLETE_SYNTHETIC = self.CLASSES_TO_PALLETE.copy()
        self.UNMAPPED_CLASS_IDX = []
        for j, (key,color)  in enumerate(self.CLASSES_TO_PALLETE.items()):
            if key in UNMAPPED_CLASSES:
                self.UNMAPPED_CLASS_IDX.append(j)

        self.UNMAPPED_CLASS_IDX = np.array(self.UNMAPPED_CLASS_IDX)
        self.color_map_synth = np.array(
            list(self.CLASSES_TO_PALLETE_SYNTHETIC.values()))\
                .astype(np.uint8)
    
    def _load_item(self, data:dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    
        file_name = data['file_name']
        mask_path  = data['mask_path']
        if 'synthetic' in data.keys():
            img_path = os.path.join(file_name)
            ann_path = os.path.join(mask_path)

        camera_images = np.array(Image.open(img_path)).astype(np.uint8)
        if self.segmentation:
            object_masks = np.array(Image.open(ann_path)).astype(np.uint8)
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
    

if __name__ == '__main__':
    config = omegaconf.OmegaConf.load('carla/carla_config.yaml')
    SEGMENTATION = True
    IMAGE_META_DATA = True
 

    # Create the dataloader and test the number of images
    dataset = CARLADataset(config, image_meta_data=IMAGE_META_DATA,
                            segmentation=SEGMENTATION)

    dataset[1935]
    # try except
    try:
        collate_fn = functools.partial(
            collate_fn, 
            segmentation=SEGMENTATION,
            image_meta_data=IMAGE_META_DATA
        )
        torch.manual_seed(0)
        dataloader = DataLoader(
            dataset, 
            batch_size=10,
            shuffle=True, 
            collate_fn=collate_fn
        )
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)
        print(data[0].shape)
    except Exception as e:
        print(e)