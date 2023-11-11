# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import *


from mmengine.dataset import Compose
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS
from waymo_open_data.data_loader import WaymoDataset
from mmcv.transforms import Resize
from mmcv.image.geometric import _scale_size
import mmcv
from omegaconf import DictConfig
import numpy as np
import os
@TRANSFORMS.register_module()
class AVResize(Resize):
    """Resize images & bbox & seg & keypoints & instance & object.
    
    Wrapper on mmcv.imresize to resize instance maps and object maps

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)
    - gt_instance_map (optional)
    - gt_object_map (optional)
    

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypois
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float,
                                                           float]]] = None,
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation='bilinear') -> None:
        super().__init__(scale, scale_factor, keep_ratio, clip_object_border,
                         backend, interpolation)


    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('gt_seg_map', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_map'] = gt_seg

        if results.get('gt_instance_map', None) is not None:
            if self.keep_ratio:
                gt_instance = mmcv.imrescale(
                    results['gt_instance_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_instance = mmcv.imresize(
                    results['gt_instance_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_instance_map'] = gt_instance
        
        if results.get('gt_object_map', None) is not None:
            if self.keep_ratio:
                gt_object = mmcv.imrescale(
                    results['gt_object_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_object = mmcv.imresize(
                    results['gt_object_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_object_map'] = gt_object
        
        return results

@DATASETS.register_module()
class WaymoDatasetMM(BaseSegDataset):
    def __init__(self, 
                data_config:Dict,
                validation=False,
                segmentation= True,
                image_meta_data=False,
                lazy_init: bool=False,
                indices:Optional[Union[int, Sequence[int]]] =None,
                serialize_data:bool = True,
                reduce_zero_label: bool = False,
                pipeline:List[Union[dict, Callable]]=None,
                max_refetch: int = 1000) -> None:
        
        # Create the dataset config from the WaymoDataset config

        self.waymo_config = DictConfig(
            {
                "TRAIN_DIR": data_config["TRAIN_DIR"],
                "EVAL_DIR": data_config["EVAL_DIR"],
                "TEST_SET_SOURCE": data_config["TEST_SET_SOURCE"],
                "SAVE_FRAMES": data_config["SAVE_FRAMES"],
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
        self._metainfo.update(
            dict(
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label))
        # self._metainfo.update(
        #     dict(
        #         label_map=self.label_map))
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))
        self._fully_initialized = False
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = validation
        if not lazy_init:
            self.full_init()

        if validation:
            assert self._metainfo.get('classes') is not None, \
                'dataset metainfo `classes` should be specified when testing'
    
    

        self.pipeline = Compose(pipeline)
        
    def waymo_init(self, 
                   segmentation: bool = True,
                   validation: bool = False,
                   image_meta_data: bool = False) -> None:
        if segmentation:
            if validation:
                self.FOLDER = self.waymo_config.EVAL_DIR
                self.contexts_path = os.path.join(self.FOLDER, '2d_pvps_validation_frames.txt')
            else:
                self.FOLDER = self.waymo_config.TRAIN_DIR
                self.contexts_path = os.path.join(self.FOLDER, '2d_pvps_training_frames.txt')
        else:
            if validation:
                self.FOLDER = self.waymo_config.EVAL_DIR
                self.contexts_path = os.path.join(self.FOLDER, '2d_detection_validation_metadata.txt')
            else:
                self.FOLDER = self.waymo_config.TRAIN_DIR
                self.contexts_path = os.path.join(self.FOLDER, '2d_detection_training_metadata.txt')

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
                context_name = line.strip().split(',')[0]
                context_frame = int(line.strip().split(',')[1])
                if not self.segmentation:
                    camera_ids = []
                    available_camera_ids = line.strip().split(',')[2:-1]
                    available_camera_ids = [int(x) - 1 for x in available_camera_ids]  
                    common_camera_ids = list(
                        set(available_camera_ids
                            .intersection(
                                set(self.waymo_config.SAVE_FRAMES)
                                )
                            )
                    )
                    
                    self._num_images += len(common_camera_ids)

                    for camera_id in common_camera_ids:
                        data_info = {
                            'context_name': context_name,
                            'context_frame': context_frame,
                            'camera_id': camera_id
                        }
                        self.data_list.append(data_info)
                        
                else:
                    for camera_id in self.waymo_config.SAVE_FRAMES:
                        data_info = {
                            'context_name': context_name,
                            'context_frame': context_frame,
                            'camera_id': camera_id
                        }
                        self.data_list.append(data_info)
                    self._num_images+=len(self.waymo_config.SAVE_FRAMES)
        
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
        
        # Useful for downstream synthesis
        self.UNMAPPED_CLASSES = ['undefined', 'ego_vehicle', 'dynamic', 'static','ground',
                                 'other_large_vehicle',   'trailer',
                                 'pedestrian_object', 'cyclist', 
                                 'motorcyclist', 'construction_cone_pole','ground_animal',
                                 'lane_marker', 'road_marker', 'sign','bird']
        
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
                context_name=data_dict['context_name'],
                context_frame=data_dict['context_frame'],
                camera_id=data_dict['camera_id'],
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
            )
            data_list.append(data_info)
            
        return data_list

            
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_dict = self.get_data_info(idx)
        if self.waymo_ds.segmentation:
            if self.waymo_ds.image_meta_data:
                images, sem_masks, instance_masks, object_masks, image_metadata = self.waymo_ds[idx]
                data_dict['img'] = images
                data_dict['gt_seg_map'] = object_masks
                data_dict['gt_instance_map'] = instance_masks
                data_dict['gt_object_map'] = sem_masks
                for key, value in image_metadata.items():
                    data_dict[key] = value
                
            else:     
                images, sem_masks, instance_masks, object_masks = self.waymo_ds[idx]
                data_dict['img'] = images
                data_dict['gt_seg_map'] = object_masks
                data_dict['gt_instance_map'] = instance_masks
                data_dict['gt_object_map'] = sem_masks
                
                
        elif not self.waymo_ds.segmentation:
            if self.waymo_ds.image_meta_data:
                images, labels, boxes, image_metadata = self.waymo_ds[idx]
                data_dict['img'] = images
                data_dict['gt_bboxes'] = boxes
                data_dict['gt_labels'] = labels
                for key, value in image_metadata.items():
                    data_dict[key] = value
                
            else:
                images, labels, boxes = self.waymo_ds[idx]
                data_dict['img'] = images
                data_dict['gt_bboxes'] = boxes
                data_dict['gt_labels'] = labels
        
        data_dict = self.pipeline(data_dict)       
        return data_dict
    
if __name__ == '__main__':
    raise NotImplementedError