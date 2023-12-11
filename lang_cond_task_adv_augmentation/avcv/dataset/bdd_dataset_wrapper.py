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
            "TRAIN_DIR": data_config["TRAIN_DIR"],
            "VAL_DIR": data_config["VAL_DIR"],
            "SYNTH_TRAIN_DIR": data_config["SYNTH_DIR"],
            "TRAIN_META_PATH": data_config["TRAIN_META_PATH"],
            "VAL_META_PATH": data_config["VAL_META_PATH"],
            "PALLETE": data_config["PALLETE"], # COCO or BDD or ADE20K or ADE20K_COCO
        }
        )
        
        self.mixing_ratio = mixing_ratio
        self.bdd_init(segmentation=segmentation,
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
                if os.path.existsos.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv"):
                    self.metadata_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv")
                self.contexts_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                        "filenames_val_seg.txt")
            else:
                if os.path.existsos.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv"):
                    self.metadata_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv")
                self.contexts_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                        "filenames_val_det.txt")
        elif not validation:
            if segmentation:
                if os.path.existsos.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv"):
                    self.metadata_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv")
                self.contexts_path = os.path.join(self.bdd_config.TRAIN_META_PATH,
                                        "filenames_train_seg.txt")
            else:
                if os.path.existsos.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv"):
                    self.metadata_path = os.path.join(self.bdd_config.VAL_META_PATH,
                                                "metadata_val_seg.csv")
                self.contexts_path = os.path.join(self.bdd_config.TRAIN_META_PATH,
                                        "filenames_train_det.txt")
        
        added_images = 0
        self.init_pallete()
        while added_images < self.synth_data_length:
            with open(self.contexts_path, 'r') as f:
                
                for line in f:
                    data_info = {
                        'file_name': line.strip(),
                    }
                    self.data_list.append(data_info)
                    self._num_images+=1
                    added_images+=1


    
# What I want to do is I can specify what Pallete mapping I need, hence if I use
# ADE20K, I can map the classes to the ADE20K classes, and if I use COCO, I can map
# the classes to the COCO classes. Or ADE20K_COCO which means I can map to ADE20K preferably
#prior to mapping it to COCO. This is useful for downstream synthesis.