# Wrtie the dataloader corresponding to the saved camera images and segmentation images and image language descriptons

# from waymo_open_dataset.protos import camera_segmentation_metrics_pb2 as metrics_pb2
# from waymo_open_dataset.protos import camera_segmentation_submission_pb2 as submission_pb2
# from waymo_open_dataset.wdl_limited.camera_segmentation import camera_segmentation_metrics

from waymo_open_data_parser.parser import *
import pickle
from torch.utils.data import Dataset, DataLoader
import omegaconf
from tqdm import tqdm
from multiprocess import Pool
from typing import *
import numpy as np
import os
from waymo_open_dataset.utils import camera_segmentation_utils
import torch 
import functools

def image_mask_pickler(config, validation=False):
    '''
    Extracts the image, and image segmentation mask from the files as an np array
      and pickles the data files
    for a dataloader to use directly downstream

    Args:
        config: The config file for the project (omegaconf)
    
    Returns:
        None

    '''
    context_set = set()
    
    # read the text file 2d_pvps_training_frames.txt and extract the context names
    if validation:
        FOLDER = config.EVAL_DIR
    else:
        FOLDER = config.TRAIN_DIR
    with open(os.path.join(FOLDER, '2d_pvps_training_frames.txt'), 'r') as f:
        for line in f:
            context_name = line.strip().split(',')[0]
            context_set.add(context_name)
    
    context_list = list(context_set)
    assert len(context_set) == 696, "The data is unbalanced, please check the data"

    # Create the folders to save the data
    if not os.path.exists(os.path.join(FOLDER, 'camera/')):
        os.makedirs(os.path.join(FOLDER, 'camera/'))
    
    if not os.path.exists(os.path.join(FOLDER, 'segmentation/')):
        os.makedirs(os.path.join(FOLDER, 'segmentation/'))

    if not os.path.exists(os.path.join(FOLDER, 'instance/')):
        os.makedirs(os.path.join(FOLDER, 'instance/'))
    
    context_pooler = Pool(processes=config.NUM_CPU)

    configs = [config]*len(context_list)
    idx = [j for j in range(len(context_list))]
    total_contexts = [len(context_list)]*len(context_list)
    validation = [validation]*len(context_list)

    # FOR TESTING ONLY
    # inputs = zip(configs[:config.NUM_CPU], context_list[:config.NUM_CPU],
    #               idx[:config.NUM_CPU], total_contexts[:config.NUM_CPU], 
    #               validation[:config.NUM_CPU])

    inputs = zip(configs, context_list, idx, total_contexts, validation)

    context_pooler.starmap(load_context_list, inputs)
    context_pooler.close()
    context_pooler.join()
   
    
    # camera_images_list = []
    # semantic_masks_lists = []
    # instance_masks_list = []

    # print('Unpacking the results')
    # for result in tqdm(results.get()):
    #     if result[0] is None or result[1] is None or result[2] is None:
    #         continue
    #     camera_images_list += result[0]
    #     semantic_masks_lists += result[1]
    #     instance_masks_list += result[2]
    # print('Done unpacking the results')
        # Pickle the data

    

def load_context_list(config, context_name, idx, total_contexts, validation):
    '''
    Loads the data for a single context

    Args:
        config: The config file for the project (omegaconf)
        context_name: The name of the context to load
        idx: The index of the context in the list of contexts
        total_contexts: The total number of contexts
        validation: Whether the context is for validation or not
    
    Returns:
        
    '''

    print("Loading context: {} of {} Name: {} Validation: {}".format(idx, 
                                                                    total_contexts,
                                                                      context_name, 
                                                                      validation))
    
    if validation:
        FOLDER = config.EVAL_DIR
    else:
        FOLDER = config.TRAIN_DIR

    try:
        frames_with_seg, camera_images = load_data_set_parquet(config=config, 
                                                                context_name=context_name, 
                                                                validation=validation)

        semantic_labels_multiframe, \
            instance_labels_multiframe, \
            panoptic_labels = read_semantic_labels(config,frames_with_seg)
        
        camera_images_frame = read_camera_images(config, camera_images)
    except:
        print("Failed context: {} of {} Name: {} Validation: {}".format(idx, total_contexts,
                                                                      context_name, 
                                                                      validation))
        return
    
    print("Finished context: {} of {} Name: {} Validation: {}".format(idx, total_contexts,
                                                                      context_name, 
                                                                      validation))
    # Save it in a pickle file
    image_index = 0
    for j, (images, semantic_masks, instance_masks) in enumerate(zip(camera_images_frame,
                                                    semantic_labels_multiframe, 
                                                    instance_labels_multiframe)):
        for frame_id in config.SAVE_FRAMES:
            with open(os.path.join(FOLDER, 'camera/camera_images{}_{}.pkl'.\
                    format(context_name, image_index)), 'wb') as f:
                pickle.dump(images[frame_id], f)

            with open(os.path.join(FOLDER, 'segmentation/semantic_masks{}_{}.pkl'\
                    .format(context_name, image_index)), 'wb') as f:
                pickle.dump(semantic_masks[frame_id], f)
            
            with open(os.path.join(FOLDER, 'instance/instance_masks{}_{}.pkl'\
                    .format(context_name, image_index)), 'wb') as f:
                pickle.dump(instance_masks[frame_id], f)
            image_index += 1

    print("Finished context: {} of {} Name: {} Validation: {}".format(idx, 
                                                                    total_contexts,
                                                                      context_name, 
                                                                      validation))
    return

class WaymoDataset(Dataset):
    '''
    Loads the dataset from the pickled files
    '''
    CLASSES: str
    PALLETE: str
    FOLDER: str
    # camera_files: List[str]
    # segment_files: List[str]
    # instance_files: List[str]
    num_images: int
    context_set: Set[str]
    segment_frames: Dict[str, List[str]]
    ds_length: int
    ds_config: omegaconf
    segmentation: bool
    image_meta_data: bool # Whether to return the image meta data or not when accessing items

    def __init__(self, config, 
                 validation=False, 
                 segmentation=True,
                 image_meta_data=False) -> None:
        super().__init__()

        if segmentation:
            if validation:
                self.FOLDER = config.EVAL_DIR
                self.contexts_path = os.path.join(self.FOLDER, '2d_pvps_validation_frames.txt')
            else:
                self.FOLDER = config.TRAIN_DIR
                self.contexts_path = os.path.join(self.FOLDER, '2d_pvps_training_frames.txt')
        else:
            if validation:
                self.FOLDER = config.EVAL_DIR
                self.contexts_path = os.path.join(self.FOLDER, '2d_detection_validation_metadata.txt')
            else:
                self.FOLDER = config.TRAIN_DIR
                self.contexts_path = os.path.join(self.FOLDER, '2d_detection_training_metadata.txt')

        self.context_set = set()
        self.segment_frames = dict()
        self.num_images = 0
        self.ds_config = config
        self.validation = validation
        self.segmentation = segmentation
        self.image_meta_data = image_meta_data

        with open(self.contexts_path, 'r') as f:
            for line in f:
                context_name = line.strip().split(',')[0]
                context_frame = int(line.strip().split(',')[1])
                self.context_set.add(context_name)
                if self.segment_frames.get(context_name) is None:
                    self.segment_frames[context_name] = [context_frame]
                else:
                    self.segment_frames[context_name].append(context_frame)
                self.num_images += 1
                
        self.num_images *= len(config.SAVE_FRAMES)
        # assert len(self.camera_files) == len(self.segment_files)\
        #       == len(self.instance_files), \
        #     "The number of files in the camera, segmentation and instance folders \
        # are not equal"

        # self.num_images = len(self.camera_files)
        # # Find the number of images

        # RGB colors used to visualize each semantic segmentation class.
        
        self.CLASSES_TO_PALLETTE = {
            'undefined' : [0, 0, 0],
            'ego_vehicle': [102, 102, 102],
            'car': [0, 0, 142], 
            'truck': [0, 0, 70], 
            'bus': [0, 60, 100],
            'other_large_vehicle': [61, 133, 198],
            'bicycle': [119, 11, 32],
            'motorcycle': [0, 0, 230],
            'trailer': [111, 168, 220],
            'pedestrian': [220, 20, 60],
            'cyclist': [255, 0, 0],
            'motorcyclist': [180, 0, 0],
            'bird': [127, 96, 0],
            'ground_animal': [91, 15, 0],
            'construction_cone_pole': [230, 145, 56],
            'pole': [153, 153, 153],
            'pedestrian_object': [234, 153, 153],
            'sign': [246, 178, 107],
            'traffic_light': [250, 170, 30],
            'building': [70, 70, 70],
            'road': [128, 64, 128],
            'lane_marker': [234, 209, 220],
            'road_marker': [217, 210, 233],
            'sidewalk': [244, 35, 232],
            'vegetation': [107, 142, 35],
            'sky': [70, 130, 180],
            'ground': [102, 102, 102],
            'dynamic': [102, 102, 102],
            'static': [102, 102, 102]
        }

        self.CLASSES = list(self.CLASSES_TO_PALLETTE.keys())
        self.PALLETE = list(self.CLASSES_TO_PALLETTE.values())
        self.color_map = np.array(self.PALLETE).astype(np.uint8)

    def __len__(self) -> int:
        # return max(len(self.camera_files), 
        #             len(self.segment_files), 
        #             len(self.instance_files))
        return self.num_images

    def get_text_description(self, object_mask: np.ndarray) -> str:
        '''
        Returns the text description of the object mask

        Args:
            object_mask: The object mask to extract the text description from
        
        Returns:
            text_description: The text description of the object mask
        '''
        
        object_set = set(object_mask.flatten().tolist())
        text_description = ''
        for object in object_set:
            text_description += self.CLASSES[object] + ', '
        return text_description[:-1]
    
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

    def __getitem__(
            self, 
            index:int
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

        # open the semantic label and instance label files
        #  and return the dataset
        # camera_file = os.path.join(self.FOLDER, self.camera_files[index])
        # segment_file = os.path.join(self.FOLDER, self.segment_files[index])
        # instance_file = os.path.join(self.FOLDER, self.instance_files[index])

        # with open(camera_file, 'rb') as f:
        #     camera_images = pickle.load(f)
        # with open(segment_file, 'rb') as f:
        #     semantic_masks = pickle.load(f)
        # with open(instance_file, 'rb') as f:
        #     instance_masks = pickle.load(f)
        
        # object_masks = self.get_object_class(semantic_masks)

        if index >= self.num_images:
            raise IndexError("Index out of range")

        # Find the appropriate index at which the image is stored
        index_copy = index
        cum_sum = 0
        for k in self.segment_frames.keys():
            cum_sum += len(self.segment_frames[k])*len(self.ds_config.SAVE_FRAMES)
            if cum_sum>=index:
                context_name = k
                len_context = len(self.segment_frames[k])*len(self.ds_config.SAVE_FRAMES)
                context_frame = self.segment_frames[k][int((index - cum_sum + len_context)\
                                                           /len(self.ds_config.SAVE_FRAMES))]
                camera_id = self.ds_config.SAVE_FRAMES[(index - cum_sum + len_context\
                                                        )%len(self.ds_config.SAVE_FRAMES)]
                break
        # Load all the frames from the context file

        if self.segmentation:
            frames_with_seg, camera_images = load_data_set_parquet(
                config=self.ds_config, 
                context_name=context_name, 
                validation=self.validation,
                context_frames=[context_frame]
            )

            semantic_labels_multiframe, \
            instance_labels_multiframe, \
            panoptic_labels = read_semantic_labels(
                self.ds_config,
                frames_with_seg
            )
            
            camera_images_frame = read_camera_images(
                self.ds_config,
                camera_images
            )

            # All semantic labels are in the form of object indices defined by the PALLETE
            camera_images = camera_images_frame[0][camera_id]
            object_masks = semantic_labels_multiframe[0][camera_id].astype(np.int64)
            instance_masks = instance_labels_multiframe[0][camera_id].astype(np.int64)

            semantic_mask_rgb = self.get_semantic_mask(object_masks)
            panoptic_mask_rgb = camera_segmentation_utils.panoptic_label_to_rgb(object_masks,
                                                                               instance_masks)
        else:
            boxes, camera_images = load_data_set_parquet(
                config=self.ds_config, 
                context_name=context_name, 
                validation=self.validation,
                context_frames=[context_frame],
                segmentation=False
            )

            box_classes, bounding_boxes = read_box_labels(
                self.ds_config,
                boxes
            )
            
            camera_images_frame = read_camera_images(
                self.ds_config,
                camera_images
            )

            camera_images = camera_images_frame[0][camera_id]
            box_classes = box_classes[camera_id][0]
            bounding_boxes = bounding_boxes[camera_id][0]
        
        img_data = {
            'context_name': context_name,
            'context_frame': context_frame,
            'camera_id': camera_id
        }

        if self.segmentation:
            if self.image_meta_data:
                return camera_images, semantic_mask_rgb, instance_masks, object_masks, img_data
            else:
                return camera_images, semantic_mask_rgb, instance_masks, object_masks
        else:
            if self.image_meta_data:
                return camera_images, box_classes, bounding_boxes, img_data
            else:
                return camera_images, box_classes, bounding_boxes

def waymo_collate_fn(data, segmentation=False, image_meta_data=False):

    if not segmentation and not image_meta_data:
        images, labels, boxes = zip(*data)
        images = torch.stack(images, 0)
        return images, labels, boxes
    elif not segmentation and image_meta_data:
        images, labels, boxes, img_data = zip(*data)
        images = torch.stack(images, 0)
        return images, labels, boxes, img_data
    elif segmentation and not image_meta_data:
        images, sem_masks, instance_masks, object_masks = zip(*data)
        images = torch.stack(images, 0)
        sem_masks = torch.stack(sem_masks, 0)
        instance_masks = torch.stack(instance_masks, 0)
        object_masks = torch.stack(object_masks, 0)
        return images, sem_masks, instance_masks, object_masks
    else:
        images, sem_masks, instance_masks, object_masks, img_data = zip(*data)
        images = torch.stack(images, 0)
        sem_masks = torch.stack(sem_masks, 0)
        instance_masks = torch.stack(instance_masks, 0)
        object_masks = torch.stack(object_masks, 0)
        return images, sem_masks, instance_masks, object_masks, img_data

if __name__ == '__main__':
    config = omegaconf.OmegaConf.load('waymo_open_data_parser/config.yaml')
    SEGMENTATION = False
    IMAGE_META_DATA = False
    if config.SAVE_DATA:
        # Append the cwd to the paths in the config file
        # for keys in config.keys():
        #     if 'DIR' in keys:
        #         config[keys] = os.path.join(os.getcwd(), config[keys])

        # Pickle the data
        image_mask_pickler(config, validation=False)
    else:
        # Create the dataloader and test the number of images
        dataset = WaymoDataset(config, image_meta_data=IMAGE_META_DATA,
                                segmentation=SEGMENTATION)

        # try except
        try:
            collate_fn = functools.partial(
                waymo_collate_fn, 
                segmentation=SEGMENTATION,
                image_meta_data=IMAGE_META_DATA
            )
            
            dataloader = DataLoader(
                dataset, 
                batch_size=10,
                shuffle=True, 
                collate_fn=collate_fn
            )
            
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)
            print(data[0].shape)
        except:
            raise ValueError("The dataloader failed to load,try creating a custom dataloader with your own collate_fn ")
