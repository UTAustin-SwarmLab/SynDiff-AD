# Wrtie the dataloader corresponding to the saved camera images and segmentation images and image language descriptons

from waymo_open_dataset.protos import camera_segmentation_metrics_pb2 as metrics_pb2
from waymo_open_dataset.protos import camera_segmentation_submission_pb2 as submission_pb2
from waymo_open_dataset.wdl_limited.camera_segmentation import camera_segmentation_metrics
from waymo_open_dataset.utils import camera_segmentation_utils

from waymo_open_data_parser.parser import *
import pickle
from torch.utils.data import Dataset, DataLoader
import omegaconf
from tqdm import tqdm
from multiprocess import Pool

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
    with open(os.path.join(config.TRAIN_DIR, '2d_pvps_training_frames.txt'), 'r') as f:
        for line in f:
            context_name = line.strip().split(',')[0]
            context_set.add(context_name)
    
    context_list = list(context_set)
    assert len(context_set) == 696, "The data is unbalanced, please check the data"
    
    context_pooler = Pool(processes=config.NUM_CPU)

    configs = [config]*len(context_list)
    idx = [j for j in range(len(context_list))]
    total_contexts = [len(context_list)]*len(context_list)
    validation = [validation]*len(context_list)

    inputs = zip(configs[:32], context_list[:32], idx[:32], total_contexts[:32], validation[:32])

    results = context_pooler.starmap_async(load_context_list, inputs)
    context_pooler.join()
    context_pooler.close()
    
    camera_images_list = []
    semantic_masks_lists = []
    instance_masks_list = []

    print('Unpacking the results')
    for result in tqdm(results.get()):
        if result[0] is None or result[1] is None or result[2] is None:
            continue
        camera_images_list += result[0]
        semantic_masks_lists += result[1]
        instance_masks_list += result[2]
    print('Done unpacking the results')
        # Pickle the data
    with open(config.data_dir + 'camera_images_list.pkl', 'wb') as f:
        pickle.dump(camera_images_list, f)

    with open(config.data_dir + 'semantic_masks_lists.pkl', 'wb') as f:
        pickle.dump(semantic_masks_lists, f)
    
    with open(config.data_dir + 'instance_masks_list.pkl', 'wb') as f:
        pickle.dump(instance_masks_list, f)
    

def load_context_list(config, context_name, idx, total_contexts, validation):
    '''
    Loads the data for a single context
    '''

    print("Loading context: {} of {} Name: {} Validation: {}".format(idx, 
                                                                    total_contexts,
                                                                      context_name, 
                                                                      validation))
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
        return None, None, None
    
    print("Finished context: {} of {} Name: {} Validation: {}".format(idx, total_contexts,
                                                                      context_name, 
                                                                      validation))
     
    return camera_images_frame, semantic_labels_multiframe, instance_labels_multiframe

class WaymoDataset(Dataset):
    '''
    Loads the dataset from the pickled files
    '''
    def __init__(self, config, validation=False) -> None:
        super().__init__()
        if validation:
            self.image_path = os.path.join(config.EVAL_DIR,'camera_images_list.pkl')
            self.semantic_path = os.path.join(config.EVAL_DIR,'semantic_masks_lists.pkl')
            self.instance_path = os.path.join(config.EVAL_DIR,'instance_masks_list.pkl')
        else:
            self.image_path = os.path.join(config.TRAIN_DIR,'camera_images_list.pkl')
            self.semantic_path = os.path.join(config.TRAIN_DIR,'semantic_masks_lists.pkl')
            self.instance_path = os.path.join(config.TRAIN_DIR,'instance_masks_list.pkl')

        with open(self.image_path, 'rb') as f:
            self.camera_images_list = pickle.load(f)
        
        with open(self.semantic_path, 'rb') as f:
            self.semantic_masks_lists = pickle.load(f)

        with (open(self.instance_path, 'rb')) as f:
            self.instance_masks_list = pickle.load(f)
        

    def __len__(self) -> int:
        return max(len(self.camera_images_list), 
                    len(self.semantic_mask_list), 
                    len(self.instance_masks_list))
    

    def __getitem__(self, index) -> Any:
        
        camera_images = self.camera_images_list[index]
        semantic_masks = self.semantic_masks_lists[index]
        instance_masks = self.instance_masks_list[index]

        return camera_images, semantic_masks, instance_masks
    


if __name__ == '__main__':
    config = omegaconf.OmegaConf.load('waymo_open_data_parser/config.yaml')

    # Append the cwd to the paths in the config file
    for keys in config.keys():
        if 'DIR' in keys:
            config[keys] = os.path.join(os.getcwd(), config[keys])

    # Pickle the data
    image_mask_pickler(config, validation=False)

    # Create the dataloader and test the number of images
    dataset = WaymoDataset(config)