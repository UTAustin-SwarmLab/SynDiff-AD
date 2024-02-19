'''
    This is the class that would generate a synthetic dataset from a given dataset.
    Based on input test conditions, and a characterization of the scores of a model on the test conditions
    we create a synthetic dataset that would be used to finetune a model that would be robust to the test conditions.
'''
import os
import pandas as pd
import omegaconf
import cv2
from lang_data_synthesis.imgseg_synthesis import ImageSynthesis
import numpy as np
from waymo_open_data import WaymoDataset
import torch
from copy import deepcopy
from lang_data_synthesis.utils import write_to_csv_from_dict
from argparse import ArgumentParser
import tensorflow as tf
# from mmseg.registry import DATASETS, TRANSFORMS, MODELS
# from avcv.dataset.dataset_wrapper import *
# from mmengine.registry import init_default_scope
# from tqdm import tqdm
# init_default_scope('mmseg')
from bdd100k.data_loader import BDD100KDataset
from PIL import Image
from lang_data_synthesis.dataset import ExpDataset
from multiprocessing import Process
import multiprocessing
from carla.data_loader import CARLADataset
import time 

class SyntheticAVGenerator:
    
    def __init__(
        self,
        source_dataset,
        config,
        carla_image_bounds = None # Only for CARLA
        ) -> None:

        self.config = config
        self.dataset = source_dataset
        # Load where the data is stored
        # if self.config.SYN_DATASET_GEN.segmentation == "waymo":
        #     self.FOLDER = config.TRAIN_DIR
        #     self.contexts_path = os.path.join(self.FOLDER, '2d_pvps_training_frames.txt')
        # else:
        #     self.FOLDER = config.TRAIN_DIR
        #     self.contexts_path = os.path.join(self.FOLDER, '2d_detection_training_metadata.txt')    
        self.carla_image_bounds = carla_image_bounds
        if isinstance(self.dataset, WaymoDataset):
            self.dataset_type = "waymo"
        elif isinstance(self.dataset, BDD100KDataset):
            self.dataset_type = "bdd"
        elif isinstance(self.dataset, CARLADataset):
            self.dataset_type = "carla"
        
        if self.config.SYN_DATASET_GEN.class_eq:
            self.config.SYN_DATASET_GEN.dataset_path = \
                self.config.SYN_DATASET_GEN.dataset_path.replace("_ft","_ft_ceq")
        
        if not os.path.exists(self.config.SYN_DATASET_GEN.dataset_path):
            os.makedirs(self.config.SYN_DATASET_GEN.dataset_path)
            if self.dataset_type != "carla":
                os.makedirs(os.path.join(self.config.SYN_DATASET_GEN.dataset_path,"img"))
                os.makedirs(os.path.join(self.config.SYN_DATASET_GEN.dataset_path,"mask"))
            
        VAL_FILENAME = self.config.ROBUSTIFICATION.val_file_path
        
        # For now we compute the probabilites based the conditions in the validation set
        # TODO: switch between trraining and validation
        if os.path.exists(VAL_FILENAME):
            self.metadata_conditions = pd.read_csv(VAL_FILENAME)
            print(self.metadata_conditions.columns)
            self.dataset_length = len(self.metadata_conditions)
            print(self.metadata_conditions\
                .groupby(['condition']).size() / self.dataset_length * 100)
        
            self.grouped_df = self.metadata_conditions.groupby(['condition'])
            self.source_probability = self.metadata_conditions\
                .groupby(['condition']).size() / self.dataset_length
            self.source_probability = self.source_probability.to_dict()
            
            self.source_probability = {
                k:self.config.SYN_DATASET_GEN.source_prob_soft*v
                for k,v in self.source_probability.items()
            }
            self.source_probability = {
                k:np.exp(v)/np.sum(np.exp(list(self.source_probability.values())))
                for k,v in self.source_probability.items()
            }
            
            self.target_probability = {
                k:self.config.SYN_DATASET_GEN.target_prob_soft/v 
                for k,v in self.source_probability.items()
            }
            
            self.target_probability = {
                k:np.exp(v)/np.sum(np.exp(list(self.target_probability.values())))
                for k,v in self.target_probability.items()
            }
            self.conditions = list(self.source_probability.keys())
        else:
            if self.dataset_type != "carla":
                raise ValueError("Validation file does not exist")
        
        # Replace meta data conditions with those in the training dataset
        TRAIN_FILENAME = self.config.ROBUSTIFICATION.train_file_path
        if os.path.exists(TRAIN_FILENAME) and not self.dataset.validation:
            self.metadata_conditions = pd.read_csv(TRAIN_FILENAME)
            print(self.metadata_conditions.columns)
            self.dataset_length = len(self.metadata_conditions)
            print(self.metadata_conditions\
                .groupby(['condition']).size() / self.dataset_length * 100)
        
            self.grouped_df = self.metadata_conditions.groupby(['condition'])
            #self.conditions = list(self.grouped_df.groups.keys())
        
        if self.config.SYN_DATASET_GEN.class_eq:
            CLASS_METADATA_FILENAME = self.config.SYN_DATASET_GEN.class_meta_data
            if os.path.exists(CLASS_METADATA_FILENAME):
                self.class_metadata = pd.read_csv(CLASS_METADATA_FILENAME)
                self.classes_dataset = self.dataset.CLASSES
                print(self.class_metadata.columns)
                # Get per class occurences
                print(self.class_metadata[self.classes_dataset].sum(axis=0)/ self.dataset_length)
                
                # Merge the class metadata with the dataset metadata
                self.metadata_all = pd.merge(
                    self.class_metadata,
                    self.metadata_conditions,
                )
                self.metadata_all_grouped = self.metadata_all.groupby(['condition'])
                print(self.metadata_all.columns)
                
                    
        if self.dataset_type == "carla":
            self.metadata_conditions = pd.DataFrame.from_dict(self.dataset.METADATA)
            self.dataset_length = len(self.metadata_conditions)
            print(self.metadata_conditions\
                .groupby(['condition']).size() / self.dataset_length * 100)
            self.grouped_df = self.metadata_conditions.groupby(['condition'])
            self.conditions = list(self.grouped_df.groups.keys())
            
        if self.dataset_type != "carla":
             
            if self.config.SYN_DATASET_GEN.segmentation:
                
                self.metadata_path = os.path.join(
                    self.config.SYN_DATASET_GEN.dataset_path,
                    "metadata_seg_{}.csv".format(self.config.seed_offset)
                )
                self.txtfilename = os.path.join(
                    self.config.SYN_DATASET_GEN.dataset_path,
                    "filenames_seg_{}.txt".format(self.config.seed_offset)
                )
            else:
                self.metadata_path = os.path.join(
                    self.config.SYN_DATASET_GEN.dataset_path,
                    "metadata_det.csv"
                )
                self.txtfilename = os.path.join(
                    self.config.SYN_DATASET_GEN.dataset_path,
                    "filenames_det.txt"
                )
            
            dict_data = {'filename':'filename', 'condition':'condition'}
            write_to_csv_from_dict(
                dict_data = dict_data, 
                csv_file_path=self.metadata_path, 
                file_name=""
            )
            
        self.prompt_df = None
        if self.config.SYN_DATASET_GEN.use_llava_prompt:
            self.llavafilename = os.path.join(self.config.SYN_DATASET_GEN.llava_prompt_path)
            self.prompt_df = pd.read_csv(self.llavafilename)

        # Load the model for synthesis
   
        self.synthesizer = ImageSynthesis(config)
    
    def sample_source_image(self, source_condition):  
        # Get a random index corresponding to the source condition
        if not self.config.SYN_DATASET_GEN.class_eq:
            source_index = self.grouped_df.get_group(source_condition).sample().index[0]
            source_data = self.metadata_conditions.iloc[source_index].to_dict()
        else:
            # Sample a class
            sub_data_group = self.metadata_all_grouped.get_group(source_condition)
            # Choose class
            bool = False
            while not bool:
                class_index = np.random.choice(
                    np.arange(len(self.classes_dataset))
                )
                semantic_class = self.classes_dataset[class_index]
                sub_data = sub_data_group[sub_data_group[semantic_class] == 1]
                if len(sub_data) > 0:
                    bool = True
            
            # Sample random index from sub_data
            idx = np.random.choice(
                np.arange(len(sub_data))
            )
            data = sub_data.iloc[idx].to_dict()
            
            # sub_data_grouped = sub_data.groupby(['condition'])
            # source_group = sub_data_grouped.get_group(source_condition)
            # source_index = source_group.sample().index[0]
            # data = source_group.iloc[source_index==source_group.index].to_dict()
            source_data = dict()
            for k,v in data.items():
                if k in self.classes_dataset:
                    if k == semantic_class:
                        #assert list(v.values())[0] == 1
                        assert v == 1
                else:
                    #source_data[k] =list(v.values())[0]
                    source_data[k] = v
            # Sample a class from the list of source conditions
            
            
        # Get the data from the source index
        # source_data = self.metadata_conditions.iloc[source_index]
        return source_data
    
    def generate_prompt(self, prompt, source_condition, target_condition):
        if self.dataset_type == "waymo" or self.dataset_type == "bdd":
            
            weather, day = target_condition.split(',') 
            source_weather, source_day = source_condition.split(',')
            
            prompt = prompt.replace(source_weather, weather)
            prompt = prompt.replace(source_day, day)
            # remove all the words associated with the source condition

            prompt = prompt.replace(source_day.lower(), day)   
            
            if 'y' in source_weather:
                prompt = prompt.replace(' '+source_weather[:-1].lower(), ' '+weather)
                if 'y' in weather:
                    prompt = prompt.replace(source_weather[:-1], weather[:-1])
                else:
                    prompt = prompt.replace(source_weather[:-1], weather)
            else:
                prompt = prompt.replace(source_weather.lower(), weather)
        elif self.dataset_type == "carla":
            prompt = prompt.replace(source_condition, target_condition)
        
        return prompt
            
            
    def generate_synthetic_image(self, source_data, target_condition):
           
        camera_images, _, _,\
        object_masks, img_data = self.dataset._load_item(source_data)
        
        prompt_tokens = {}
        prompt_tokens['a_prompt'] = "Must contain " + ExpDataset.get_text_description(
            object_masks,
            self.dataset.CLASSES
        )
        #prompt_tokens['n_prompt'] = "Must not contain " + self.dataset.get_text_description(object_masks)
        prompt_tokens['n_prompt'] = ""
        semantic_mapped_rgb_mask = self.dataset.get_mapped_semantic_mask(object_masks)
        invalid_mask = self.dataset.get_unmapped_mask(object_masks)
        
        if self.config.SYN_DATASET_GEN.use_llava_prompt:
            
            if self.dataset_type == "waymo":
                prompt = self.prompt_df.loc[
                    (img_data['context_name'] == self.prompt_df['context_name']) &
                    (img_data['context_frame'] == self.prompt_df['context_frame']) &
                    (img_data['camera_id'] == self.prompt_df['camera_id'])
                ]['caption'].values[0]
                source_condition = self.metadata_conditions.loc[
                    (img_data['context_name'] == self.metadata_conditions['context_name']) &
                    (img_data['context_frame'] == self.metadata_conditions['context_frame']) &
                    (img_data['camera_id'] == self.metadata_conditions['camera_id'])
                ]['condition'].values[0]
            elif self.dataset_type == "bdd":
                prompt = self.prompt_df.loc[
                    (img_data['file_name'] == self.prompt_df['context_name'])
                ]['caption'].values[0]
                source_condition = self.metadata_conditions.loc[
                    (img_data['file_name'] == self.metadata_conditions['context_name'])
                ]['condition'].values[0]
            elif self.dataset_type == 'carla':
                source_condition = img_data['condition']
                prompt = self.prompt_df.loc[
                    (img_data['route'] == self.prompt_df['route'])&
                    (img_data['file_name'] == self.prompt_df['file_name'])&
                    (img_data['mask_path'] == self.prompt_df['mask_path'])&
                    (img_data['synthetic'] == self.prompt_df['synthetic'])&
                    (img_data['condition'] == self.prompt_df['condition'])
                ]['caption'].values[0]
                

            if isinstance(target_condition, list):
                p = deepcopy(prompt)
                prompt_list = []
                for condition in target_condition:
                    p = deepcopy(prompt)    
                    prompt_list.append(self.generate_prompt(p, source_condition, condition))
                prompt = prompt_list
            elif isinstance(target_condition, str):
                prompt = self.generate_prompt(prompt, source_condition, target_condition)
                
            #source_condition = self.metadata_conditions.iloc[source_idx]['condition']
            # if self.dataset_type == "waymo" or self.dataset_type == "bdd":
                
            #     if isinstance(target_condition, list):
                    
            #     else:
            #         source_weather, source_day = source_condition.split(',')
                    
            #         prompt = prompt.replace(source_weather, weather)
            #         prompt = prompt.replace(source_day, day)
                    
            #         if 'context_name' in img_data.keys() or 'file_name' in img_data.keys():
            #             # remove all the words associated with the source condition

            #             prompt = prompt.replace(source_day.lower(), day)   
                        
            #             if 'y' in source_weather:
            #                 prompt = prompt.replace(' '+source_weather[:-1].lower(), ' '+weather)
            #                 prompt = prompt.replace(source_weather[:-1], weather)
            #             else:
            #                 prompt = prompt.replace(source_weather.lower(), weather)
            # elif self.dataset_type == "carla":   
            #     if isinstance(target_condition, list):
            #         prompt_list = []
            #         for condition in target_condition:
            #             p = deepcopy(prompt)    
            #             prompt_list.append(p.replace(source_condition, condition))
            #         prompt = prompt_list
            #     elif isinstance(target_condition, str):
            #         prompt = prompt.replace(source_condition, target_condition)
        else:
            if self.dataset_type == "waymo" or self.dataset_type == "bdd":
                if isinstance(target_condition, list):
                    prompt = []
                    for condition in target_condition:
                        weather, day = condition.split(',')
                        p = 'This image is taken during {} time of the day and\
                        features {} weather. '.format(day, weather)
                        prompt.append(p)
                elif isinstance(target_condition, str):
                    weather, day = target_condition.split(',')
                    prompt = 'This image is taken during {} time of the day and\
                    features {} weather. '.format(day, weather)
                
            elif self.dataset_type == "carla":
                if isinstance(target_condition, list):
                    prompt_list = []
                    for condition in target_condition:
                        p = 'This image is taken during {} weather and day condition.'.format(condition)
                        prompt_list.append(p)
                    prompt = prompt_list
                elif isinstance(target_condition, str):
                    prompt = 'This image is taken during {} weather and day condition.'.format(target_condition)
        
        with torch.no_grad():
            outputs = self.synthesizer.run_model(
                camera_images.copy(), 
                seg_mask=[semantic_mapped_rgb_mask.copy()],
                prompt = prompt,
                prompt_tokens=prompt_tokens
            )
        
        results = outputs[1:]
        for j,syn_img in enumerate(results):
            syn_img = cv2.resize(syn_img, (camera_images.shape[1],
                                           camera_images.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
            if not self.config.SYN_DATASET_GEN.use_finetuned:
                syn_img[invalid_mask.squeeze()] = camera_images[invalid_mask.squeeze()]
                
            # For carla we ensure the traffic lights are road lines are preserved from original image image            if self.dataset_type == "carla":
            if self.dataset_type == "carla":
                syn_img[invalid_mask.squeeze()] = camera_images[invalid_mask.squeeze()]
                
            outputs[j+1] = syn_img
        
        # Return the synthetic image, object mask 
        
        return outputs[1:], object_masks, img_data
        
    def prepare_source_target(self, synth_image_id):
        
        if self.dataset_type == "waymo" or self.dataset_type == "bdd":
            np.random.seed(synth_image_id*100 + self.config.seed_offset)
            # Sample a source conditions from the test conditions
            source_condition_idx= np.random.choice(
                np.arange(len(self.conditions)),
                p=list(self.source_probability.values())
            )
            source_condition = self.conditions[source_condition_idx]
            t_cond_prob = deepcopy(list(self.target_probability.values()))
            #t_cond_prob[source_condition_idx] = 0
            t_cond_prob = [p/sum(t_cond_prob) for p in t_cond_prob]
            # Sample a target condition from the test conditions
            target_condition = np.random.choice(self.conditions,
                                                p=t_cond_prob,
                                                size = self.config.SYNTHESIS_PARAMS.NUMSAMPLES)
            target_condition = target_condition.tolist()

            dataset_info = self.sample_source_image(source_condition)
        elif self.dataset_type == "carla":
            np.random.seed((synth_image_id +  self.carla_image_bounds[0])*100 + self.config.seed_offset)
            
            source_condition = self.dataset.METADATA[synth_image_id + self.carla_image_bounds[0]]['condition']
            #t_cond_prob[source_condition_idx] = 0
            # Sample a target condition from the test conditions
            target_condition = np.random.choice(self.conditions,
                                                size = self.config.SYNTHESIS_PARAMS.NUMSAMPLES,
                                                )
            target_condition = target_condition.tolist()

            dataset_idx = synth_image_id + self.carla_image_bounds[0]
            dataset_info = self.dataset.METADATA[dataset_idx]
        
        return dataset_info, source_condition, target_condition
    
    def generate_synthetic_dataset(self):
        '''
            This function would generate a synthetic dataset based on the test conditions
            and the model's performance on the test conditions.
        '''
        if self.dataset_type == "waymo" or self.dataset_type == "bdd":   
            num_images = self.config.SYN_DATASET_GEN.num_synthetic_images
        elif self.dataset_type == "carla":
            num_images = self.carla_image_bounds[1] - self.carla_image_bounds[0]
        for j in range(num_images):
            
            dataset_idx, source_condition, target_condition = self.prepare_source_target(j)
            
            print("Iteration {} : Source condition:{} Target: {} Idx :{}".format(j,
                                                                          source_condition,
                                                                          target_condition,
                                                                          dataset_idx))
    
            # Generate a synthetic image
            images, obj_mask, img_data = self.generate_synthetic_image(dataset_idx,
                                                             target_condition)
            
            # Save the synthetic image
            for i,image in enumerate(images):
                

                if self.dataset_type == "waymo":
                    
                    file_name = str(img_data['context_name'])+"_"\
                    +str(img_data['context_frame'])+"_"+\
                    str(img_data['camera_id'])+"_"+\
                    str(j)+"_"+str(i)+"_"+str(self.config.seed_offset)
                    
                elif self.dataset_type == "bdd":
                    
                   file_name = str(img_data['file_name'])+\
                    "_"+ str(j)+"_"+str(i)+"_"+str(self.config.seed_offset)
                    
                    
                    
                if self.dataset_type == "waymo" or self.dataset_type == "bdd":
                    cv2.imwrite(os.path.join(self.config.SYN_DATASET_GEN.dataset_path,
                                            "img",
                                            file_name +".png"), image)
                    #  save the synthetic mask
                    im = Image.fromarray(obj_mask.astype(np.uint8).squeeze())
                    save_im_path = os.path.join(
                        self.config.SYN_DATASET_GEN.dataset_path, 
                        "mask", 
                        file_name+".png"
                    )
                    im.save(save_im_path)
                    # # Save the synthetic mask as a numpy array
                    
                    # Write the synthetic metadata with the target condition in a csv file
                
                    dict_data = {'filename':file_name, 'condition':target_condition}
                    write_to_csv_from_dict(
                        dict_data = dict_data, 
                        csv_file_path=self.metadata_path, 
                        file_name=""
                    )
                                    # Write the filename in a txt file
                    with open(self.txtfilename, 'a') as f:
                        f.write(file_name+"\n")                
                
                elif self.dataset_type == "carla":
                    
                    # Note that the route folder name is modified based on the source folder
                    # for the synthetic image, this allows us to generate different variations
                    # of the route without having to modify the ground truth semantic content
                    # of the route and the associated control and plan commands. 
                    # We dont store the copy metadata for carla precisely for this reason.
                    synthetic_route = "synth___"+ img_data['route'] + "___v{}".format(i)
                    
                    synth_route_path = os.path.join(
                        self.config.SYN_DATASET_GEN.dataset_path,
                        synthetic_route
                    )
                    
                    if not os.path.exists(synth_route_path):
                        os.makedirs(synth_route_path)
                    
                    img_path = img_data['file_name']
                    #img_path = img_path.replace(img_data['route'], synthetic_route)
                    source_folder_name = os.path.join(
                        img_path.split(img_data['route'])[0],
                        img_data['route']
                    )
                                                      
                    img_subpath = img_path.split(img_data['route'])[1].split("/")[1]
                    
                    if not os.path.exists(os.path.join(synth_route_path, img_subpath)):
                        os.makedirs(os.path.join(synth_route_path, img_subpath))
                        
                    img_path = img_path.replace(source_folder_name, synth_route_path)
                    
                    cv2.imwrite(img_path, image)
                    #  save the synthetic mask

def parse_args():
    parser = ArgumentParser(description='Image Classification with CLIP')

    parser.add_argument(
        '--validation',
        action='store_true',
        default=False,
        help='Use validation dataset')
    parser.add_argument(
        '--segmentation',
        action='store_true',
        default=False,
        help='Enable it for the segmentation dataset')
    parser.add_argument(
        '--img_meta_data',
        action = 'store_true',
        default = False,
        help = 'Enable it for the image meta data dataset')
         
    parser.add_argument(
        '--experiment',
        choices=['waymo', 'bdd', 'carla', 'cliport'],
        default='none',
        help='Which experiment config to generate data for')
    
    parser.add_argument(
        '--seed_offset',
        type=int,
        default=2000,
        help='Offset seed for random number generators')
    parser.add_argument(
        '--num_process',
        type = int,
        default=1,
        help='How many parallel processes' 
    )
    parser.add_argument(
        '--class_eq',
        default=False,
        action='store_true',
        help='Whether to sample classes equally'
    )
    return  parser.parse_args()


def process(args, worker_id = None):
    SEGMENTATION = args.segmentation
    IMAGE_META_DATA = True
    VALIDATION = False
    
    tf.config.set_visible_devices([], 'GPU')
    
    config_file_name = 'lang_data_synthesis/{}_config.yaml'.format(args.experiment)
    config = omegaconf.OmegaConf.load(config_file_name)

    if worker_id is not None:
        torch.cuda.set_device('cuda:{}'.format(worker_id))
        config.seed_offset = int(100/args.num_process)*worker_id + 1
    else:
        config.seed_offset = args.seed_offset
        
    if args.experiment == 'waymo':
        dataset = WaymoDataset(config.IMAGE.WAYMO, 
                            image_meta_data=IMAGE_META_DATA,
                            segmentation=SEGMENTATION,
                            validation=VALIDATION)
    elif args.experiment == 'bdd':
        dataset = BDD100KDataset(config.IMAGE.BDD, 
                            image_meta_data=IMAGE_META_DATA,
                            segmentation=SEGMENTATION,
                            validation=VALIDATION)
    elif args.experiment == 'carla':
        dataset = CARLADataset(config,
                            image_meta_data=IMAGE_META_DATA,
                            segmentation=SEGMENTATION,
                            validation=VALIDATION)
    elif args.experiment == 'cliport':
        raise NotImplementedError
    else:
        raise ValueError("Experiment not supported")
    
    if args.experiment == 'waymo' or args.experiment == 'bdd':
        config.SYN_DATASET_GEN.class_eq = args.class_eq
        dataset_gen = SyntheticAVGenerator(
            source_dataset=dataset,
            config=config
        )
    elif args.experiment == 'carla':
        if args.class_eq:
            raise ValueError("Class eq not supported for carla")
        config.SYN_DATASET_GEN.class_eq = False
        if worker_id is None:
            worker_id = 0
            
        if worker_id == args.num_process - 1:
            carla_image_bounds = [worker_id*int(len(dataset)/args.num_process), 
                                  len(dataset)]
        else:
            carla_image_bounds = [worker_id*int(len(dataset)/args.num_process), 
                                  (worker_id+1)*int(len(dataset)/args.num_process)]
        dataset_gen = SyntheticAVGenerator(
            source_dataset=dataset,
            config=config,
            carla_image_bounds=carla_image_bounds
        )
    dataset_gen.generate_synthetic_dataset()
    
if __name__ == "__main__":
    args = parse_args()
    

    if args.num_process == 1:
        process(args)
    else:
        multiprocessing.set_start_method('spawn', force=True)
        num_workers = args.num_process
        workers = [Process(target=process, 
                        args=(args, 
                              i))
                for i in range(num_workers)]
        
        for w in workers:
            w.start()
            time.sleep(1.0)
            

        # Collect results
        # all_captions = []
        # for _ in range(len(self.dataloader)):
        #     all_captions.extend(result_queue.get())

        # Wait for all worker processes to finish
        for w in workers:
            w.join()
