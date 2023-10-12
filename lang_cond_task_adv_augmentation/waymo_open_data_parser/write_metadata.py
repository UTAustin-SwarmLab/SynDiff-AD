'''
    Creates additional metadata for the Waymo datset in terms of the number of frames
    per context for both segmentation level and camera level files
'''

import omegaconf
import os
from typing import *
from parser import read
import pandas as pd

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

def create_metadata(config:omegaconf):
    '''
    Creates additional metadata for the Waymo datset in terms of the number of frames
    per context for both segmentation level and camera level files
    '''
    # Get the contexts
    context_set = set()
    with open(os.path.join(config.TRAIN_DIR, '2d_pvps_training_frames.txt'), 'r') as f:
        for line in f:
            context_name = line.strip().split(',')[0]
            context_set.add(context_name)
    # Get the number of frames per context for segmentation and camera images
    num_frames_seg = []
    num_frames_cam = []
    for context in context_set:
        num_frames_seg.append(get_num_frames(config,
                                            'camera_segmentation',
                                            context,
                                            ['key.segment_context_name', 
                                             'key.frame_timestamp_micros']))
        num_frames_cam.append(get_num_frames(config, 
                                             'camera_image', 
                                             context,
                                             ['key.segment_context_name', 
                                              'key.frame_timestamp_micros']))
    
    # Create a dataframe with the contexts and number of frames per context
    metadata = pd.DataFrame({'context':list(context_set), 'num_frames_seg':num_frames_seg,
                              'num_frames_cam':num_frames_cam})
    metadata.to_csv(os.path.join(config.TRAIN_DIR, 'metadata.csv'), index=False)

if __name__=="__main__":
    config = omegaconf.OmegaConf.load('waymo_open_data_parser/config.yaml')
    create_metadata(config)
