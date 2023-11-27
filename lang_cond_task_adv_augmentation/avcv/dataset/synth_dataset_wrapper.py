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
import pandas as pd

@DATASETS.register_module()
class SynthWaymoDatasetMM(BaseSegDataset):
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
                max_refetch: int = 1000) -> None:
        
        # Create the dataset config from the WaymoDataset config

        self.waymo_config = DictConfig(
            {
                "DATASET_DIR": data_config["DATASET_DIR"],
            }
        )
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
        
        
        #self.context_set = set()
        self.data_list = []
        #self.segment_frames = dict()
        self._num_images = 0
        self.validation = validation
        self.segmentation = segmentation
        self.image_meta_data = image_meta_data
        #self.context_count = dict()
        
        with open(self.contexts_path, 'r') as f:
            for line in f:
                data_info = {
                    'file_name': line.strip(),
                }
                self.data_list.append(data_info)
                self._num_images+=1
        # Set
        # Get a list of GPU devices
        
        self.CLASSES_TO_PALLETTE = {
            'undefined' : [0, 0, 0],#1
            'ego_vehicle': [102, 102, 102],#2
            'car': [0, 0, 142], #3
            'truck': [0, 0, 70], #4
            'bus': [0, 60, 100],#5
            'other_large_vehicle': [61, 133, 198],#
            'bicycle': [119, 11, 32],#
            'motorcycle': [0, 0, 230],#
            'trailer': [111, 168, 220],#
            'pedestrian': [220, 20, 60],#10
            'cyclist': [255, 0, 0],#
            'motorcyclist': [180, 0, 0],#
            'bird': [127, 96, 0],#
            'ground_animal': [91, 15, 0],#
            'construction_cone_pole': [230, 145, 56],#15
            'pole': [153, 153, 153],#
            'pedestrian_object': [234, 153, 153],#
            'sign': [246, 178, 107],#
            'traffic_light': [250, 170, 30],#
            'building': [70, 70, 70],#20
            'road': [128, 64, 128],#
            'lane_marker': [234, 209, 220],#
            'road_marker': [217, 210, 233],#
            'sidewalk': [244, 35, 232],#
            'vegetation': [107, 142, 35],#25
            'sky': [70, 130, 180],#
            'ground': [102, 102, 102],#
            'dynamic': [102, 102, 102],#
            'static': [102, 102, 102]#
        }

        self.CLASSES = list(self.CLASSES_TO_PALLETTE.keys())
        self.PALLETE = list(self.CLASSES_TO_PALLETTE.values())
        self.color_map = np.array(self.PALLETE).astype(np.uint8)
        
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
        
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_dict = self.get_data_info(idx)
        data_dict = self._waymo_get_item(data_dict)    
        data_dict = self.pipeline(data_dict)      
         
        return data_dict
    
    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')
    
if __name__ == '__main__':
    raise NotImplementedError