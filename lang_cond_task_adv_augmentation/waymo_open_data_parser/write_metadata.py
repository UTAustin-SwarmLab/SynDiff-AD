'''
    Creates additional metadata for the Waymo datset in terms of the number of frames
    per context for both segmentation level and camera level files
'''

import omegaconf
import os
from typing import *
from parser import read
import pandas as pd
from waymo_open_dataset import v2


def get_num_frames(config:omegaconf,
                          tag:str,
                          context_name:str,
                          frame_keys:List[str]):
    '''
    Creates additional metadata for the Waymo datset in terms of the number of frames
    per context for both segmentation level and camera level files
    '''

    # Load the dataset
    df = read(config, tag, context_name)
    
    frame_keys = ['key.segment_context_name', 'key.frame_timestamp_micros']
    df_merged = df.groupby(
        frame_keys, group_keys=False).agg(list)
    # Get the number of frames per context
    num_frames = len(list(df.iterrows()))
    
    return num_frames

def create_metadata_detection(config:omegaconf,
                    validation = False):
    '''
    Creates additional metadata for the Waymo datset in terms of the number of frames
    per context for both segmentation level and camera level files

    Args:
        config: Config file
        segmentation: Boolean to indicate whether to create metadata for segmentation
                      or camera images
    '''
    # Get the contexts
    context_set = set()

    # Assumes accces to the segmentation contexts which can be downloaded
    #  from the waymo poend datset demos
    if validation:
        FOLDER = config.EVAL_DIR
        file  = '2d_pvps_validation_frames.txt'
        write_file = '2d_detection_validation_metadata.txt'
    else:
        FOLDER = config.TRAIN_DIR
        file = '2d_pvps_training_frames.txt'
        write_file = '2d_detection_training_metadata.txt'

    # List all the files in the folder and get the context names and TOS
    context_set = set()
    with open(os.path.join(FOLDER, file), 'r') as f:
        for line in f:
            context_name = line.strip().split(',')[0]
            context_set.add(context_name)

    context_set = list(context_set)

    with open(write_file, 'w') as f:
        for contexts in context_set:
            # Read the camera images and segmentation files
            cam_images_df = read(config, 'camera_image', contexts, validation=validation)
            cam_boxes_df = read(config, 'camera_box', contexts, validation=validation)
            merged_df = v2.merge(cam_images_df,cam_boxes_df, right_group=True)

            # Get the number of frames per context for segmentation and camera images
            timestamp_micros = merged_df['key.frame_timestamp_micros'].unique()
            # Convert to a list
            timestamp_micros = list(timestamp_micros)

            # Write the context name and number of frames to a file
            for timestamp in timestamp_micros:
                f.write(contexts + ',' + str(timestamp) + '\n')
    f.close()
if __name__=="__main__":
    config = omegaconf.OmegaConf.load('waymo_open_data_parser/config.yaml')
    create_metadata_detection(config)
    create_metadata_detection(config, validation=True)
