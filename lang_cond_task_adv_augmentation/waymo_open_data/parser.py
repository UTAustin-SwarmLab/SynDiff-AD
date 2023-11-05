import omegaconf
import pandas as pd
import os

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import immutabledict
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing as mp
import numpy as np
import dask.dataframe as dd
import io
import PIL.Image as Image

if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import v2

from waymo_open_dataset.utils import camera_segmentation_utils

def load_dataset_proto(config: omegaconf, validation=False) -> List:
    '''
    Load the dataset and select frames that have the segmentation labels from tf proto files

    Args:
        config: OmegaConf DictConfig object with the data directory and file name (see config,yaml)
    
    Returns:
        frames_with_seg: list of frames that have the segmentation labels
    '''
    if validation:
        dataset = tf.data.TFRecordDataset(config.EVAL_DIR, compression_type='')
    else:
        dataset = tf.data.TFRecordDataset(config.TRAIN_DIR, compression_type='')
    frames_with_seg = []
    sequence_id = None
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        # Save frames which contain CameraSegmentationLabel messages. We assume that
        # if the first image has segmentation labels, all images in this frame will.
        if frame.images[0].camera_segmentation_label.panoptic_label:
            frames_with_seg.append(frame)
            if sequence_id is None:
                sequence_id = frame.images[0].camera_segmentation_label.sequence_id
            # Collect 3 frames for this demo. However, any number can be used in practice.
            if frame.images[0].camera_segmentation_label.sequence_id != sequence_id :
                continue
    
    return frames_with_seg


def organise_segment_data(config: omegaconf, frames_with_seg: list) -> List:
    '''
    Organsie the images by the order of the cameras that took the images

    Args:
        config: OmegaConf DictConfig object with the data directory and file name (see config,yaml)
        frames_with_seg: List of frames that have the segmentation labels'

    Returns:
        segmentation_protos_ordered: List of segmentation labels ordered by the camera order
    '''
    # Organize the segmentation labels in order from left to right for viz later.
    camera_left_to_right_order = [open_dataset.CameraName.SIDE_LEFT,
                              open_dataset.CameraName.FRONT_LEFT,
                              open_dataset.CameraName.FRONT,
                              open_dataset.CameraName.FRONT_RIGHT,
                              open_dataset.CameraName.SIDE_RIGHT]
    segmentation_protos_ordered = []
    for frame in frames_with_seg:
        segmentation_proto_dict = {image.name : image.camera_segmentation_label for image in frame.images}
        segmentation_protos_ordered.append([segmentation_proto_dict[name] for name in camera_left_to_right_order])
    
    return segmentation_protos_ordered


def read(config,  tag: str, context_name: str, validation=False) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    if validation:
        paths = f'{config.EVAL_DIR}/{tag}/{context_name}.parquet'
    else:
        paths = f'{config.TRAIN_DIR}/{tag}/{context_name}.parquet'
    
    try:
        df = dd.read_parquet(paths)
    except:
        raise ValueError(f'Could not read {paths}')
    return df


def ungroup_row(key_names: Sequence[str],
                key_values: Sequence[str],
                row: dd.DataFrame) -> Iterator[Dict[str, Any]]:
    """Splits a group of dataframes into individual dicts."""
    keys = dict(zip(key_names, key_values))
    cols, cells = list(zip(*[(col, cell) for col, cell in row.items()]))
    for values in zip(*cells):
        yield dict(zip(cols, values), **keys)

def load_data_set_parquet(
        config: omegaconf, 
        context_name: str,
        validation=False,
        context_frames:List = None,
        segmentation = True) -> Tuple[List[v2.CameraImageComponent], List[Any]]:
    '''
    Load datset from parquet files for segmentation and camera images
    
    Args:
        config: OmegaConf DictConfig object with the data directory and file name (see config,yaml)

    Returns:
     
       cam_segmentation_list: List of segmentation labels ordered by the camera order
    '''

    
    cam_images_df = read(config, 'camera_image', context_name, validation=validation)

    if segmentation:
        cam_segmentation_df = read(config, 'camera_segmentation', 
                                   context_name,  
                                   validation=validation)
        merged_df = v2.merge(cam_images_df,cam_segmentation_df, right_group=True)
    else:
        cam_boxes_df = read(config, 'camera_box', 
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
    for i, (key_values, r) in enumerate(cam_labels_per_frame_df.iterrows()):
        # Read three sequences of 5 camera images for this demo.
        # Store a segmentation label component for each camera.
        if segmentation:
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

    # TODO: need to figure out what the function is to obtain camera images
    # for i, (key_values, r) in enumerate(cam_images_per_frame_df.iterrows()):
    #     # Read three sequences of 5 camera images for this demo.
    #     # Store a segmentation label component for each camera.
    #     cam_list.append(
    #         [v2.CameraSegmentationLabelComponent.from_dict(d) 
    #         for d in ungroup_row(frame_keys, key_values, r)])
        
    return cam_labels_list, image_list


def read_semantic_labels(
        config: omegaconf, 
        segments: List[v2.CameraSegmentationLabelComponent]
        ) -> Tuple[List,List,List]:
    ''' 
    The dataset provides tracking for instances between cameras and over time.
    By setting remap_to_global=True, this function will remap the instance IDs in
     each image so that instances for the same object will have the same ID between
     different cameras and over time.

    Args:
        config: omega congif gfrom the config.yaml file
        segmentation_protos_ordered: List of segmentation labels ordered by the camera order
    
    Returns:
        panoptic_labels: List of panoptic labels
        semantic_labels_multiframe: List of semantic labels
        instance_labels_multiframe: List of instance labels

    '''
    segmentation_protos_flat = sum(segments, [])
    panoptic_labels, num_cameras_covered, is_tracked_masks, panoptic_label_divisor =\
          camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_segmentation_labels(
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
            semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
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

def read_box_labels(
        config: omegaconf, 
        box_labels: List[v2.CameraBoxComponent]
        ) -> Tuple[List,List,List]:
    ''' 
    The dataset provides tracking for instances between cameras and over time.
    By setting remap_to_global=True, this function will remap the instance IDs in
     each image so that instances for the same object will have the same ID between
     different cameras and over time.

    Args:
        config: omega congif gfrom the config.yaml file
        segmentation_protos_ordered: List of segmentation labels ordered by the camera order
    
    Returns:
        box_class: List of object types (classes)
        bounding_boxes: List of bounding boxes
    '''
    # We can further separate the semantic and instance labels from the panoptic
    # labels.
    NUM_CAMERA_FRAMES = 5
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


def read_camera_images(config: omegaconf, 
                        camera_images: List[v2.CameraImageComponent]
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
   
    for i in range(0, len(camera_images)):
        camera_images_frame = [[] for _ in range(NUM_CAMERA_FRAMES)]
        for j in range(len(camera_images[i])):
            cam_id = camera_images[i][j].key.camera_name - 1
            camera_images_frame[cam_id] = np.array(Image.open(
                io.BytesIO(camera_images[i][j].image)))
        camera_images_all.append(camera_images_frame)
    return camera_images_all

