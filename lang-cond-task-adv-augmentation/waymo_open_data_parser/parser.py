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

if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import v2

from waymo_open_dataset.utils import camera_segmentation_utils

def load_dataset_proto(config: omegaconf) -> List:
    '''
    Load the dataset and select frames that have the segmentation labels from tf proto files

    Args:
        config: OmegaConf DictConfig object with the data directory and file name (see config,yaml)
    
    Returns:
        frames_with_seg: list of frames that have the segmentation labels
    '''
    dataset = tf.data.TFRecordDataset(config.FILE_NAME, compression_type='')
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


def organise_segment_data_proto(config: omegaconf, frames_with_seg: list) -> List:
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


def read(config, context_name: str, tag: str, dataset_dir: str = EVAL_DIR) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  
  paths = f'{dataset_dir}/{tag}/{context_name}.parquet'
  return dd.read_parquet(paths)


def ungroup_row(key_names: Sequence[str],
                key_values: Sequence[str],
                row: dd.DataFrame) -> Iterator[Dict[str, Any]]:
    """Splits a group of dataframes into individual dicts."""
    keys = dict(zip(key_names, key_values))
    cols, cells = list(zip(*[(col, cell) for col, cell in row.items()]))
    for values in zip(*cells):
        yield dict(zip(cols, values), **keys)

def load_data_set_parquet(config, context_name: str) ->\
        Tuple(List[open_dataset.CameraImage], List[open_dataset.CameraSegmentationLabel]):
    '''
    Load datset from parquet files
    
    Args:
        config: OmegaConf DictConfig object with the data directory and file name (see config,yaml)

    Returns:
     
       cam_segmentation_list: List of segmentation labels ordered by the camera order
    '''
    context_name = '550171902340535682_2640_000_2660_000'



    cam_segmentation_df = read('camera_segmentation')
    cam_images_df = read('camera_images')

    # Group segmentation labels into frames by context name and timestamp.
    frame_keys = ['key.segment_context_name', 'key.frame_timestamp_micros']
    cam_segmentation_per_frame_df = cam_segmentation_df.groupby(
        frame_keys, group_keys=False).agg(list)

    cam_images_per_frame_df = cam_images_df.groupby(
        frame_keys, group_keys=False).agg(list)
    
    cam_segmentation_list = []
    for i, (key_values, r) in enumerate(cam_segmentation_per_frame_df.iterrows()):
        # Read three sequences of 5 camera images for this demo.
        # Store a segmentation label component for each camera.
        cam_segmentation_list.append(
            [v2.CameraSegmentationLabelComponent.from_dict(d) 
            for d in ungroup_row(frame_keys, key_values, r)])

    cam_list = []
    #TODO: need to figure out what the function is to obtain camera images
    for i, (key_values, r) in enumerate(cam_images_per_frame_df.iterrows()):
        # Read three sequences of 5 camera images for this demo.
        # Store a segmentation label component for each camera.
        cam_list.append(
            [v2.CameraSegmentationLabelComponent.from_dict(d) 
            for d in ungroup_row(frame_keys, key_values, r)])
        
    return cam_segmentation_list, cam_list


def read_semantic_labels(config: omegaconf, 
                         segmentation_protos_ordered: List[open_dataset.CameraSegmentationLabel]) ->\
                            Tuple(List[open_dataset.SemanticLabel],List[open_dataset.InstanceLabel],
                                  List[open_dataset.PanopticLabel]):
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
    segmentation_protos_flat = sum(segmentation_protos_ordered, [])
    panoptic_labels, num_cameras_covered, is_tracked_masks, panoptic_label_divisor =\
          camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_segmentation_labels(
        segmentation_protos_flat, remap_to_global=True
    )

    # We can further separate the semantic and instance labels from the panoptic
    # labels.
    NUM_CAMERA_FRAMES = 5
    semantic_labels_multiframe = []
    instance_labels_multiframe = []
    for i in range(0, len(segmentation_protos_flat), NUM_CAMERA_FRAMES):
        semantic_labels = []
        instance_labels = []
        for j in range(NUM_CAMERA_FRAMES):
            semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
            panoptic_labels[i + j], panoptic_label_divisor)
            semantic_labels.append(semantic_label)
            instance_labels.append(instance_label)
        semantic_labels_multiframe.append(semantic_labels)
        instance_labels_multiframe.append(instance_labels)

    return semantic_labels_multiframe, instance_labels_multiframe, panoptic_labels

if __name__ == "__main__":
    load_data('waymo_open_data_parser/config.yaml')