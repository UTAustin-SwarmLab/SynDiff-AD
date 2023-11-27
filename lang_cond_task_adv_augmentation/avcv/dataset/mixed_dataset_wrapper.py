# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import *


from mmengine.dataset import Compose
from mmengine.dataset import BaseDataset
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS
from waymo_open_data.parser import *
from mmcv.transforms import Resize
from mmcv.image.geometric import _scale_size
import mmcv
from omegaconf import DictConfig
import numpy as np
import os
import tensorflow as tf
import logging
from mmengine.logging import print_log
import avcv.dataset.utils as dataset_utils
from avcv.dataset.dataset_wrapper import WaymoDatasetMM
import pandas as pd

@DATASETS.register_module()
class MixedWaymoDatasetMM(WaymoDatasetMM):
    def __init__(self, 
                data_config:Dict,
                validation=False,
                segmentation= True,
                image_meta_data=False,
                filter_cfg: Optional[dict] = None,
                lazy_init: bool=False,
                indices:Optional[Union[int, Sequence[int]]] =None,
                serialize_data:bool = True,
                reduce_zero_label: bool = False,
                pipeline:List[Union[dict, Callable]]=None,
                mixing_ratio:float = 0.5,
                max_refetch: int = 1000) -> None:
        
        # Create the dataset config from the WaymoDataset config

        self.waymo_config = DictConfig(
            {
                "DATASET_DIR": data_config["DATASET_DIR"],
                "TRAIN_DIR": data_config["TRAIN_DIR"],
                "EVAL_DIR": data_config["EVAL_DIR"],
                "TEST_SET_SOURCE": data_config["TEST_SET_SOURCE"],
                "SAVE_FRAMES": data_config["SAVE_FRAMES"],
            }
        )
        self.mixing_ratio = mixing_ratio
        self.waymo_init(segmentation=segmentation,
                        validation=validation,
                        image_meta_data=image_meta_data)
        # self.waymo_ds= WaymoDataset(dataset_config, 
        #                             validation=validation, 
        #                             segmentation=segmentation, 
        #                             image_meta_data=image_meta_data)
        
        self.METAINFO = dict(
            classes = self.CLASSES,
            palette = self.PALLETE
        )
       
        self._metainfo = self._load_metainfo(copy.deepcopy(self.METAINFO))
        
        new_classes = self._metainfo.get('classes', None)
        self.label_map = self.get_label_map(new_classes)
        self.reduce_zero_label = reduce_zero_label
        self.max_refetch = max_refetch
        self.data_bytes: np.ndarray
        
        self._metainfo.update(
            dict(
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label))

        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))
        self._fully_initialized = False
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = validation
        
        self.pipeline = Compose(pipeline)
        if not lazy_init:
            self.full_init()
        
        

        if validation:
            assert self._metainfo.get('classes') is not None, \
                'dataset metainfo `classes` should be specified when testing'
         
    def waymo_init(self, 
                   segmentation: bool = True,
                   validation: bool = False,
                   image_meta_data: bool = False) -> None:
        
        super().waymo_init(segmentation=segmentation,
                            validation=validation,
                            image_meta_data=image_meta_data)
        
        self.synth_data_length = int(len(self.data_list)/(1 - self.mixing_ratio)) - len(self.data_list)
        
        # Need to change in accordance with the proposed synthetic dataset structure
        if segmentation:
            self.metadata_path = os.path.join(self.waymo_config.DATASET_DIR,
                                              "metadata_seg.csv")
            self.contexts_path = os.path.join(self.waymo_config.DATASET_DIR,
                                       "filenames_seg.txt")
        else:
            self.metadata_path = os.path.join(self.waymo_config.DATASET_DIR,
                                              "metadata_det.csv")
            self.contexts_path = os.path.join(self.waymo_config.DATASET_DIR,
                                       "filenames_det.txt")
        
        added_images = 0
        while added_images < self.synth_data_length:
            with open(self.contexts_path, 'r') as f:
                
                for line in f:
                    data_info = {
                        'file_name': line.strip(),
                    }
                    self.data_list.append(data_info)
                    self._num_images+=1
                    added_images+=1
     
    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
              ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True
    
    def load_data_list(self) -> List[dict]:
        """Convert the data list within waymo_ds to the format of mmseg

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        waymo_metadata = self.data_list
        for data_dict in waymo_metadata:
            if 'context_name' in data_dict.keys():
                data_info = dict(
                context_name=data_dict['context_name'],
                context_frame=data_dict['context_frame'],
                camera_id=data_dict['camera_id'],
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
            )
            else:
                data_info = dict(
                    file_name=data_dict['file_name'],
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label,
                    seg_fields=[],
                )
            data_list.append(data_info)
            
        return data_list

    def _waymo_get_item(
            self, 
            data_info: dict
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

        # if index >= self._num_images:
        #     index = self._rand_another()
            
        # context_name = self.data_list[index]['context_name']
        # context_frame = self.data_list[index]['context_frame']
        # camera_id = self.data_list[index]['camera_id']
        if 'context_name' in data_info.keys():
            return super()._waymo_get_item(data_info)
        
        img_path = os.path.join(self.waymo_config.DATASET_DIR, 
                                'img', data_info['file_name'] + '.png')
        ann_path = os.path.join(self.waymo_config.DATASET_DIR,
                                'mask', data_info['file_name'] + '.npy')
        
    
        camera_images = np.array(Image.open(img_path)).astype(np.uint8)
        if self.segmentation:
            object_masks = np.load(ann_path).squeeze()
            instance_masks = object_masks.copy()
            semantic_mask_rgb = self.get_semantic_mask(object_masks)
        else:
            raise NotImplementedError
        
        if self.segmentation:
            data_info['img'] = camera_images
            data_info['gt_seg_map'] = object_masks
            data_info['gt_instance_map'] = instance_masks
            data_info['gt_object_map'] = semantic_mask_rgb
            data_info['ori_shape'] = camera_images.shape[:2]
            #data_info['condition']
            return data_info
        
        else:
            raise NotImplementedError
            # data_info['img'] = camera_images
            # data_info['gt_bboxes'] = bounding_boxes
            # data_info['gt_labels'] = box_classes
            # data_info['ori_shape'] = camera_images.shape[:2]
            return data_info
 
    

if __name__ == '__main__':
    raise NotImplementedError