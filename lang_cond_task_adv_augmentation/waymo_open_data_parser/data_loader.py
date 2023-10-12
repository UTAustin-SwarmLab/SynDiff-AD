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

    def __init__(self, config, validation=False) -> None:
        super().__init__()

        if validation:
            self.FOLDER = config.EVAL_DIR
        else:
            self.FOLDER = config.TRAIN_DIR

        self.context_set = set()
        self.segment_frames = list()
        self.num_images = 0
        self.ds_config = config
        self.validation = validation
        with open(os.path.join(self.FOLDER, '2d_pvps_training_frames.txt'), 'r') as f:
            for line in f:
                context_name = line.strip().split(',')[0]
                context_frame = line.strip().split(',')[1]
                self.context_set.add(context_name)
                if self.segment_frames.get(context_name) is None:
                    self.segment_frames[context_name] = []
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
        self.inv_CLASSES_TO_PALLETE = {tuple(v): k for k, v in self.CLASSES_TO_PALLETTE.items()}

    def __len__(self) -> int:
        # return max(len(self.camera_files), 
        #             len(self.segment_files), 
        #             len(self.instance_files))
        return self.num_images
    

    def get_object_class(self, semantic_mask: np.ndarray) -> List[int]:
        '''
        Returns the object classes in the semantic mask

        Args:
            semantic_mask: The semantic mask to extract the object classes from
        
        Returns:
            object_classes: The object classes in the semantic mask
        '''
        
        mapped_object_mask = np.vectorize(lambda x: tuple(self.inv_CLASSES_TO_PALLETE.get(x)))(semantic_mask)
        return mapped_object_mask

    @staticmethod
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

    def __getitem__(self, index) -> Any:
        

        # open the semantic label and instance label files
        #  and return the data
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

        context_name = k
        for k in self.segment_frames.keys():
            self.cum_sum += len(self.segment_frames[k])*len(self.ds_config.SAVE_FRAMES)
            if self.cum_sum>=index:
                context_name = k
                context_frame = self.segment_frames[k][int((index - self.cum_sum)/3)]
                camera_id = self.ds_config.SAVE_FRAMES[(index - self.cum_sum)%3]
        # Load all the frames from the context file
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

        camera_images = camera_images_frame[camera_id]
        semantic_masks = semantic_labels_multiframe[camera_id]
        instance_masks = instance_labels_multiframe[camera_id]
        object_masks = self.get_object_class(semantic_masks)
        return camera_images, semantic_masks, instance_masks, object_masks
    


if __name__ == '__main__':
    config = omegaconf.OmegaConf.load('waymo_open_data_parser/config.yaml')

    if config.SAVE_DATA:
        # Append the cwd to the paths in the config file
        # for keys in config.keys():
        #     if 'DIR' in keys:
        #         config[keys] = os.path.join(os.getcwd(), config[keys])

        # Pickle the data
        image_mask_pickler(config, validation=False)
    else:
        # Create the dataloader and test the number of images
        dataset = WaymoDataset(config)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)
        print(data[0].shape)