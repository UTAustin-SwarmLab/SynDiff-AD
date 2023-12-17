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
from lang_data_synthesis.utils import ADE_20K_PALETTE, COCO_PALETTE
from bdd100k.data_loader import BDD100KDataset

@DATASETS.register_module()
class BDDDatasetMM(BaseSegDataset):
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
        
        
    
        self.bdd_config = DictConfig(
            {
                'DATASET_DIR': data_config['DATASET_DIR'],
                'SYNTH_TRAIN_DIR': data_config['SYNTH_TRAIN_DIR'], # Always set in the sub programs
                'TRAIN_META_PATH': data_config['TRAIN_META_PATH'],
                'VAL_META_PATH': data_config['VAL_META_PATH'],
                'PALLETE': data_config['PALLETE'] # COCO or BDD or ADE20K or ADE20K_COCO
            }
        )
        
        self.dataset = BDD100KDataset(config=self.bdd_config,
                                      validation=validation,
                                      segmentation=segmentation,
                                      image_meta_data=image_meta_data,
                                      mixing_ratio=mixing_ratio)

        self.METAINFO = dict(
            classes = self.dataset.CLASSES,
            palette = self.dataset.PALLETE
        )
       
        self._metainfo = self._load_metainfo(copy.deepcopy(self.METAINFO))
        
        new_classes = self.dataset.CLASSES
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
        self.color_map = np.array(self.dataset.PALLETE).astype(np.uint8)
        
        self.pipeline = Compose(pipeline)
        if not lazy_init:
            self.full_init()
        
        if validation:
            assert self._metainfo.get('classes') is not None, \
                'dataset metainfo `classes` should be specified when testing'
         
        
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
        bdd_metadata = self.dataset.METADATA
        for data_dict in bdd_metadata:
            data_info = dict(
                file_name=data_dict['file_name'],
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
            )
            data_list.append(data_info)
            
        return data_list

    def _bdd_get_item(
            self, 
            data_info: dict,
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
        
        
        if self.segmentation:
            if self.dataset.image_meta_data:
                camera_images, semantic_mask_rgb, instance_masks, object_masks, img_meta_data= \
                    self.dataset._load_item_(data_info)
            
            data_info['img'] = camera_images
            data_info['gt_seg_map'] = object_masks
            data_info['gt_instance_map'] = instance_masks
            data_info['gt_object_map'] = semantic_mask_rgb
            data_info['ori_shape'] = camera_images.shape[:2]
            return data_info
        
        else:
            raise NotImplementedError('Only segmentation is supported')
            data_info['img'] = camera_images
            data_info['gt_bboxes'] = bounding_boxes
            data_info['gt_labels'] = box_classes
            data_info['ori_shape'] = camera_images.shape[:2]
            data_info['condition'] = condition
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
        data_dict = self._bdd_get_item(data_dict)    
        data_dict = self.pipeline(data_dict)      
         
        return data_dict
    
    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum li mit of refetech is reached.

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