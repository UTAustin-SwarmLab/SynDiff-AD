from typing import *
from abc import ABC, abstractclassmethod
from torch.utils.data import Dataset
import numpy as np
import torch

class ExpDataset(Dataset):
    
    def __init__(self) -> None:
        super().__init__()
        self._num_images = 0 
        self.data_list = []
    
    
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
            mask = np.logical_or(mask, object_mask == idx)
        return mask
    
    def __len__(self) -> int:
        # return max(len(self.camera_files), 
        #             len(self.segment_files), 
        #             len(self.instance_files))
        return self._num_images


    
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
