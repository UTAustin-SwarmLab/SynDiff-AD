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
                 interpolation='bilinear',
                 test = True) -> None:
        super().__init__(scale, scale_factor, keep_ratio, clip_object_border,
                         backend, interpolation)
        self.test = test

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if self.test:
            return results 
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
                filter_cfg: Optional[dict] = None,
                lazy_init: bool=False,
                indices:Optional[Union[int, Sequence[int]]] =None,
                serialize_data:bool = True,
                reduce_zero_label: bool = False,
                pipeline:List[Union[dict, Callable]]=None,
                max_refetch: int = 1000,
                cache: bool = True) -> None:
        
        # Create the dataset config from the WaymoDataset config

        self.waymo_config = DictConfig(
            {
                "TRAIN_DIR": data_config["TRAIN_DIR"],
                "EVAL_DIR": data_config["EVAL_DIR"],
                "TEST_SET_SOURCE": data_config["TEST_SET_SOURCE"],
                "SAVE_FRAMES": data_config["SAVE_FRAMES"],
            }
        )
        self.cache = cache
        
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
        if segmentation:
            if validation:
                self.FOLDER = self.waymo_config.EVAL_DIR
                self.contexts_path = os.path.join(self.FOLDER, '2d_pvps_validation_frames.txt')
                    
            else:
                self.FOLDER = self.waymo_config.TRAIN_DIR
                self.contexts_path = os.path.join(self.FOLDER, '2d_pvps_training_frames.txt')
            
            if self.cache:
                self.cache_folder = os.path.join(self.FOLDER, "pvps_cache")

        else:
            if validation:
                self.FOLDER = self.waymo_config.EVAL_DIR
                self.contexts_path = os.path.join(self.FOLDER, '2d_detection_validation_metadata.txt')
            else:
                self.FOLDER = self.waymo_config.TRAIN_DIR
                self.contexts_path = os.path.join(self.FOLDER, '2d_detection_training_metadata.txt')
            if self.cache:
                self.cache_folder = os.path.join(self.FOLDER, "det_cache")
        
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
            
            
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
        
        context_name = data_info['context_name']
        context_frame = data_info['context_frame']
        camera_id = data_info['camera_id']
        
        file_name = str(context_name) +"_" +str(context_frame)+"_"+str(camera_id)+".npy"
        path = os.path.join(self.cache_folder, file_name)
        if self.cache:
            if os.path.exists(path):
                data_info = np.load(path)
                return data_info
        
        camera_labels, camera_images, camera_weather, camera_lighting = self.load_data_set_parquet(
                context_name=context_name, 
                validation=self.validation,
                context_frames=[context_frame]
            )
        
        if self.segmentation:
            semantic_labels_multiframe, \
            instance_labels_multiframe, \
            panoptic_labels = camera_labels
            
            # All semantic labels are in the form of object indices defined by the PALLETE
            camera_images = camera_images[0][camera_id]
            object_masks = semantic_labels_multiframe[0][camera_id].astype(np.int64)
            instance_masks = instance_labels_multiframe[0][camera_id].astype(np.int64)

            semantic_mask_rgb = self.get_semantic_mask(object_masks)
            # panoptic_mask_rgb = camera_segmentation_utils.panoptic_label_to_rgb(object_masks,
            #                                                                    instance_masks)
        else:

            box_classes, bounding_boxes = camera_labels
            camera_images = camera_images[0][camera_id]
            box_classes = box_classes[camera_id][0]
            bounding_boxes = bounding_boxes[camera_id][0]
        weather = camera_weather[0][camera_id]
        lighting_conditions = camera_lighting[0][camera_id]
        condition = weather + ', ' + lighting_conditions
        if self.segmentation:
            data_info['img'] = camera_images
            data_info['gt_seg_map'] = object_masks
            data_info['gt_instance_map'] = instance_masks
            data_info['gt_object_map'] = semantic_mask_rgb
            data_info['ori_shape'] = camera_images.shape[:2]
            data_info['condition'] = condition
            
        
        else:
            data_info['img'] = camera_images
            data_info['gt_bboxes'] = bounding_boxes
            data_info['gt_labels'] = box_classes
            data_info['ori_shape'] = camera_images.shape[:2]
            data_info['condition'] = condition
        
        if self.cache:
            if not os.path.exists(path):
                np.save(path, data_info)
                
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
    
    def load_data_set_parquet(
        self,
        context_name: str,
        validation=False,
        context_frames:List = None) -> Tuple[List[v2.CameraImageComponent], List[Any]]:
        '''
        Load datset from parquet files for segmentation and camera images
        
        Args:
            config: OmegaConf DictConfig object with the data directory and file name (see config,yaml)

        Returns:
        
        cam_segmentation_list: List of segmentation labels ordered by the camera order
        '''

    
        cam_images_df = read(self.waymo_config, 'camera_image', context_name, validation=validation)
        stats_df = read(self.waymo_config, 'stats', context_name, validation=validation)
        cam_images_df = v2.merge(cam_images_df, stats_df, right_group=True)
        if self.segmentation:
            cam_segmentation_df = read(self.waymo_config, 'camera_segmentation', 
                                    context_name,  
                                    validation=validation)
            merged_df = v2.merge(cam_images_df,cam_segmentation_df, right_group=True)
        else:
            cam_boxes_df = read(self.waymo_config, 'camera_box', 
                                context_name,
                                validation=validation)
            merged_df = v2.merge(cam_images_df,cam_boxes_df, right_group=True)

        # Group segmentation labels into frames by context name and timestamp.
        frame_keys = ['key.segment_context_name', 'key.frame_timestamp_micros']

        if context_frames is None:
            #frame_keys = ['key.segment_context_name', 'key.frame_timestamp_micros']
            cam_labels_per_frame_df = merged_df.groupby(
                frame_keys, group_keys=False).agg(list)
        else:
            
            #frame_keys = ['key.segment_context_name', 'key.frame_timestamp_micros']
            # cam_segmentation_per_frame_df = merged_df.groupby(
            #     frame_keys, group_keys=True).agg(list)
            
            # filter out the frames that are not in the context_frames
            # cam_labels_per_frame_df = merged_df.reset_index()
            cam_labels_per_frame_df = merged_df.set_index('key.frame_timestamp_micros')
            cam_labels_per_frame_df = cam_labels_per_frame_df.loc[context_frames]
            cam_labels_per_frame_df = cam_labels_per_frame_df.groupby(
                frame_keys, group_keys=False).agg(list)
            
        cam_labels_list = []
        image_list = []
        weather_list = []
        for i, (key_values, r) in enumerate(cam_labels_per_frame_df.iterrows()):
            # Read three sequences of 5 camera images for this demo.
            # Store a segmentation label component for each camera.
            if self.segmentation:
                cam_labels_list.append(
                    [v2.CameraSegmentationLabelComponent.from_dict(d) 
                    for d in ungroup_row(frame_keys, key_values, r)])
            else:
                cam_labels_list.append(
                [v2.CameraBoxComponent.from_dict(d) 
                for d in ungroup_row(frame_keys, key_values, r)])
                
            image_list.append(
                [v2.CameraImageComponent.from_dict(d) 
                for d in ungroup_row(frame_keys, key_values, r)])
            
            weather_list.append(
            [v2.StatsComponent.from_dict(d) 
             for d in ungroup_row(frame_keys, key_values, r)])
        
        parsed_cam_labels = self.read_labels(cam_labels_list)
        parsed_cam_images, parsed_weather, parsed_lightning = self.read_camera_images(image_list, weather_list)
        return parsed_cam_labels, parsed_cam_images, parsed_weather, parsed_lightning
    
    def read_labels(
        self, 
        cam_labels: Union[List[v2.CameraBoxComponent],
                    List[v2.CameraSegmentationLabelComponent]]
            ) -> Any:
        
        if self.segmentation:
            segments = cam_labels
            segmentation_protos_flat = sum(segments, [])
            panoptic_labels, num_cameras_covered, is_tracked_masks, panoptic_label_divisor =\
                dataset_utils.decode_multi_frame_panoptic_labels_from_segmentation_labels(
                segmentation_protos_flat, remap_to_global=False
            )

            # We can further separate the semantic and instance labels from the panoptic
            # labels.
            NUM_CAMERA_FRAMES = 5
            semantic_labels_multiframe = []
            instance_labels_multiframe = []
            panoptic_labels_multiframe = []
            sum_ = 0
            for i in range(0, len(segments)):
                semantic_labels = [[] for _ in range(NUM_CAMERA_FRAMES)]
                instance_labels = [[] for _ in range(NUM_CAMERA_FRAMES)]
                panoptic_labels_list = [[] for _ in range(NUM_CAMERA_FRAMES)]
                for j in range(len(segments[i])):
                    semantic_label, instance_label = \
                        camera_segmentation_utils.\
                            decode_semantic_and_instance_labels_from_panoptic_label(
                    panoptic_labels[sum_], panoptic_label_divisor)
                    cam_id = segments[i][j].key.camera_name - 1
                    semantic_labels[cam_id] = semantic_label
                    instance_labels[cam_id] = instance_label
                    panoptic_labels_list[cam_id] = panoptic_labels[sum_]
                    sum_ += 1
                semantic_labels_multiframe.append(semantic_labels)
                instance_labels_multiframe.append(instance_labels)
                panoptic_labels_multiframe.append(panoptic_labels_list)

            return semantic_labels_multiframe, instance_labels_multiframe, panoptic_labels_multiframe
        else:
            NUM_CAMERA_FRAMES = 5
            box_labels = []
            box_classes_frame = [[] for _ in range(NUM_CAMERA_FRAMES)] # For the entire frame per camera
            bounding_boxes_frame = [[] for _ in range(NUM_CAMERA_FRAMES)] # For the entire frame per camera
            try:
                for i in range(0, len(box_labels)):
                    for j in range(len(box_labels[i])):
                        # Get camera ID
                        cam_id = box_labels[i][j].key.camera_name - 1
                        box_classes_frame[cam_id].append(box_labels[i][j].type)
                        bounding_boxes_frame[cam_id].append([ np.array([
                            box_labels[i][j].box.center.x,
                            box_labels[i][j].box.center.y,
                            box_labels[i][j].box.size.x,
                            box_labels[i][j].box.size.y
                        ])])
            except:
                print('Box labels not found')
                box_classes_frame = None
                bounding_boxes_frame = None

            return box_classes_frame, bounding_boxes_frame   
    
    def read_camera_images(self, 
                        camera_images: List[v2.CameraImageComponent],
                        weather_conditions: List[v2.StatsComponent] =None,
                        return_weather_cond = True
                        ) -> List[np.ndarray]:
        '''
        Read camera images from the dataset

        Args:
            config: omega config from the config.yaml file
            camera_images: List of camera images
        
        Returns:
            camera_images: List of camera images
        '''
        NUM_CAMERA_FRAMES = 5
        camera_images_all = []
        weather_conditions_all = []
        light_conditions_all = []
        
        for i in range(0, len(camera_images)):
            camera_images_frame = [[] for _ in range(NUM_CAMERA_FRAMES)]
            weather_conditions_frame = [[] for _ in range(NUM_CAMERA_FRAMES)]
            lighting_conditions_frame = [[] for _ in range(NUM_CAMERA_FRAMES)]
            for j in range(len(camera_images[i])):
                cam_id = camera_images[i][j].key.camera_name - 1
                camera_images_frame[cam_id] = np.array(Image.open(
                    io.BytesIO(camera_images[i][j].image)))
                weather_conditions_frame[cam_id] = weather_conditions[i][j].weather
                lighting_conditions_frame[cam_id] = weather_conditions[i][j].time_of_day
                
            camera_images_all.append(camera_images_frame)
            weather_conditions_all.append(weather_conditions_frame)
            light_conditions_all.append(lighting_conditions_frame)
        if return_weather_cond:
            return camera_images_all, weather_conditions_all, light_conditions_all
     
        return camera_images_all

if __name__ == '__main__':
    raise NotImplementedError