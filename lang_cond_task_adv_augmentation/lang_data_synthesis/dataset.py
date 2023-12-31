from typing import *
from abc import ABC, abstractclassmethod
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pandas as pd
from omegaconf import OmegaConf
from ControlNet.annotator.util import resize_image
import cv2
class ExpDataset(Dataset):
    
    def __init__(self) -> None:
        super().__init__()
        self._num_images = 0 
        self._data_list = []
    
    
    @property
    def METADATA(self):
        return self._data_list
    
    
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
            if idx == 0:
                continue
            mask = np.logical_or(mask, object_mask == idx)
        return mask
    
    def __len__(self) -> int:
        # return max(len(self.camera_files), 
        #             len(self.segment_files), 
        #             len(self.instance_files))
        return self._num_images

    def _load_item(self, data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Loads the item from the dataset

        Args:
            data: The data to load the item from
        
        Returns:
            image: The image tensor
            sem_mask: The semantic mask tensor
            instance_mask: The instance mask tensor
        '''
        raise NotImplementedError
    
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
            if CLASSES[object] == 'undefined':
                continue
            text_description += CLASSES[object] + ', '
        return text_description[:-1]


def collate_fn(
        data,
        segmentation=False, 
        image_meta_data=False):

    if not segmentation and not image_meta_data:
        images, labels, boxes = zip(*data)
        images = torch.tensor(np.stack(images, 0), dtype=torch.uint8)
        return images, labels, boxes
    elif not segmentation and image_meta_data:
        images, labels, boxes, img_data = zip(*data)
        images = torch.tensor(np.stack(images, 0), dtype=torch.uint8)
        return images, labels, boxes, img_data
    elif segmentation and not image_meta_data:
        images, sem_masks, instance_masks, object_masks = zip(*data)
        images = torch.tensor(np.stack(images, 0), dtype=torch.uint8)
        sem_masks = torch.tensor(np.stack(sem_masks, 0), dtype=torch.uint8)
        instance_masks = torch.tensor(np.stack(instance_masks, 0), dtype=torch.int64)
        object_masks = torch.tensor(np.stack(object_masks, 0), dtype=torch.int64)
        return images, sem_masks, instance_masks, object_masks
    else:
        images, sem_masks, instance_masks, object_masks, img_data = zip(*data)
        images = torch.tensor(np.stack(images, 0), dtype=torch.uint8)
        sem_masks = torch.tensor(np.stack(sem_masks, 0), dtype=torch.uint8)
        instance_masks = torch.tensor(np.stack(instance_masks, 0), dtype=torch.int64)
        object_masks = torch.tensor(np.stack(object_masks, 0), dtype=torch.int64)
        return images, sem_masks, instance_masks, object_masks, img_data


class AVControlNetDataset(ExpDataset):
    
    def __init__(self, config_dict,
                image_meta_data=True,
                segmentation=True,
                validation=False) -> None:
        super().__init__()
        
        # We would write an init for both the waymo and the nuscenes datasets
        
        self.dataset = dict()
        self.prompt_df = dict()
        self.image_meta_data = image_meta_data
        self.config_dict = config_dict
        for dataset, cfg in config_dict.items():
            if dataset == 'waymo':
                from waymo_open_data.data_loader import WaymoDataset
                self.dataset[dataset] = WaymoDataset(cfg.IMAGE.WAYMO,
                            image_meta_data=image_meta_data,
                            segmentation=segmentation,
                            validation=validation)
            elif dataset == 'bdd':
                from bdd100k.data_loader import BDD100KDataset
                self.dataset[dataset] = BDD100KDataset(cfg.IMAGE.BDD,
                            image_meta_data=image_meta_data,
                            segmentation=segmentation,
                            validation=validation)
            else:
                raise ValueError(f'Invalid dataset {dataset}')
            
            prompt_file = os.path.join(cfg.SYN_DATASET_GEN.llava_prompt_path)
            self.prompt_df[dataset] = pd.read_csv(prompt_file)
        
        self._num_images = sum([len(ds) for ds in self.dataset.values()])
            
        self.av_init()
            
    def av_init(self):
        
        for ds_name,ds in self.dataset.items():
            meta = ds.METADATA
            for info in meta:
                info['dataset'] = ds_name
            
            self._data_list.extend(meta)
    
    # We need to return the prompt, segmentation image and the target image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We would write a getitem for both the waymo and the nuscenes datasets

        if idx> self._num_images:
            raise IndexError(f'Index {idx} out of range')
    
        data_info = self._data_list[idx]
        dataset = data_info['dataset']
        
        if self.image_meta_data:
            camera_images, _, _,\
            object_masks, _ = self.dataset[dataset]._load_item(data_info)
        else:
            camera_images, _, _,\
            object_masks = self.dataset[dataset]._load_item(data_info)
        
        semantic_mask = self.dataset[dataset].get_mapped_semantic_mask(object_masks)
        
        if dataset == 'waymo':
            prompt = self.prompt_df[dataset].loc[
                (data_info['context_name'] == self.prompt_df[dataset]['context_name']) &
                (data_info['context_frame'] == self.prompt_df[dataset]['context_frame']) &
                (data_info['camera_id'] == self.prompt_df[dataset]['camera_id'])
            ]['caption'].values[0]
        elif dataset == 'bdd':
            prompt = self.prompt_df[dataset].loc[
                (data_info['file_name'] == self.prompt_df[dataset]['file_name'])
            ]['caption'].values[0]
        
        RESIZE = self.config_dict[dataset].SYNTHESIS_PARAMS.IMAGE_RESOLUTION
        camera_images = cv2.resize(camera_images, (RESIZE, RESIZE), interpolation= cv2.INTER_AREA)
        #camera_images = resize_image(camera_images, RESIZE)
        H, W, C = camera_images.shape
            
        semantic_mask = cv2.resize(semantic_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        return dict(jpg=camera_images, txt=prompt, hint=semantic_mask)
        

if __name__ == "__main__":
    configs = {
        'waymo' : OmegaConf.load('lang_data_synthesis/waymo_config.yaml'),
        'bdd' : OmegaConf.load('lang_data_synthesis/bdd_config.yaml')
    }
    
    # Switch
    ds = AVControlNetDataset(configs)
    print(len(ds))
    
    print(ds[10000].keys())
    print(ds[10000]['jpg'].shape)
    print(ds[10000]['txt'])
    print(ds[10000]['hint'].shape)
    print(ds[40000].keys())
    print(ds[40000]['jpg'].shape)
    print(ds[40000]['txt'])
    print(ds[40000]['hint'].shape)
    