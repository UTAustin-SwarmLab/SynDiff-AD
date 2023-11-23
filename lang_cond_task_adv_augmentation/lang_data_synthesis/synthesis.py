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
class SyntheticAVGenerator:
    
    def __init__(
        self,
        source_dataset,
        config) -> None:

        self.config = config
        self.dataset = source_dataset
        # Load where the data is stored
        # if self.config.SYN_DATASET_GEN.segmentation == "waymo":
        #     self.FOLDER = config.TRAIN_DIR
        #     self.contexts_path = os.path.join(self.FOLDER, '2d_pvps_training_frames.txt')
        # else:
        #     self.FOLDER = config.TRAIN_DIR
        #     self.contexts_path = os.path.join(self.FOLDER, '2d_detection_training_metadata.txt')    
        
        if not os.path.exists(self.config.SYN_DATASET_GEN.dataset_path):
            os.makedirs(self.config.SYN_DATASET_GEN.dataset_path)
            os.makedirs(os.path.join(self.config.SYN_DATASET_GEN.dataset_path,"img"))
            os.makedirs(os.path.join(self.config.SYN_DATASET_GEN.dataset_path,"mask"))
            
        FILENAME = self.config.SYN_DATASET_GEN.conditions_path + "waymo_env_conditions_val.csv"

        if os.path.exists(FILENAME):
            self.metadata_conditions = pd.read_csv(FILENAME)
            print(self.metadata_conditions.columns)
            self.dataset_length = len(self.metadata_conditions)
            print(self.metadata_conditions\
                .groupby(['condition']).size() / self.dataset_length * 100)
        
            self.grouped_df = self.metadata_conditions.groupby(['condition'])
            self.source_probability = self.metadata_conditions\
                .groupby(['condition']).size() / self.dataset_length
            self.source_probability = self.source_probability.to_dict()
            
            self.target_probability = {k:self.config.SYN_DATASET_GEN.target_prob_soft/v 
                                       for k,v in self.source_probability.items()}
            
            self.target_probability = {k:np.exp(v)/np.sum(np.exp(list(self.target_probability.values())))
                                        for k,v in self.target_probability.items()}
            self.conditions = list(self.source_probability.keys())
        else:
            raise Exception("File not found")
        
        self.metadata_path = os.path.join(self.config.SYN_DATASET_GEN.dataset_path,"metadata.csv")
        dict_data = {'filename':'filename', 'condition':'condition'}
        write_to_csv_from_dict(
            dict_data = dict_data, 
            csv_file_path=self.metadata_path, 
            file_name=""
        )
        # Load the model for synthesis
        config = omegaconf.OmegaConf.load('lang_data_synthesis/config.yaml')
        self.synthesizer = ImageSynthesis(config)
    
    def sample_source_image(self, source_condition):  
        # Get a random index corresponding to the source condition
        source_index = self.grouped_df.get_group(source_condition).sample().index[0]
        # Get the data from the source index
        # source_data = self.metadata_conditions.iloc[source_index]
        return source_index
        
    def generate_synthetic_image(self, source_idx, target_condition):
           
        camera_images, _, _,\
        object_masks, img_data = self.dataset[source_idx]
        prompt_tokens = {}
        prompt_tokens['a_prompt'] = "Must contain " + self.dataset.get_text_description(object_masks)
        prompt_tokens['n_prompt'] = "Must not contain " + self.dataset.get_text_description(object_masks)
    
        semantic_mapped_rgb_mask = self.dataset.get_mapped_semantic_mask(object_masks)
        invalid_mask = self.dataset.get_unmapped_mask(object_masks)
        weather, day = target_condition.split(',')
        prompt = ' A photorealistic camera image taken during {} weather conditions during {} time'.format(weather, day)
        with torch.no_grad():
            outputs = self.synthesizer.run_model(camera_images.copy(), 
                                seg_mask=[semantic_mapped_rgb_mask.copy()],
                                prompt = prompt,
                                prompt_tokens=prompt_tokens)
        
        results = outputs[1:]
        for j,syn_img in enumerate(results):
            syn_img = cv2.resize(syn_img, (camera_images.shape[1],
                                           camera_images.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
            syn_img[invalid_mask.squeeze()] = camera_images[invalid_mask.squeeze()]
            outputs[j+1] = syn_img
        
        # Return the synthetic image, object mask 
        
        return outputs[1:], object_masks, img_data
        
        
    def generate_synthetic_dataset(self):
        '''
            This function would generate a synthetic dataset based on the test conditions
            and the model's performance on the test conditions.
        '''
        
        for j in range(self.config.SYN_DATASET_GEN.num_synthetic_images):
            np.random.seed(j*100)
            # Sample a source conditions from the test conditions
            source_condition_idx= np.random.choice(np.arange(len(self.conditions)),
                                                p=list(self.source_probability.values()))
            source_condition = self.conditions[source_condition_idx]
            t_cond_prob = deepcopy(list(self.target_probability.values()))
            t_cond_prob[source_condition_idx] = 0
            t_cond_prob = [p/sum(t_cond_prob) for p in t_cond_prob]
            # Sample a target condition from the test conditions
            target_condition = np.random.choice(self.conditions,
                                                p=t_cond_prob)

            dataset_idx = self.sample_source_image(source_condition)
            print("Iteration {} : Source condition:{} Target: {} Idx :{}".format(j,
                                                                          source_condition,
                                                                          target_condition,
                                                                          dataset_idx))
    
            # Generate a synthetic image
            images, obj_mask, img_data = self.generate_synthetic_image(dataset_idx,
                                                             target_condition)
            
            # Save the synthetic image
            for i,image in enumerate(images):
                file_name = str(img_data['context_name'])+"_"\
                    +str(img_data['context_frame'])+"_"+\
                    str(img_data['camera_id'])+"_"+\
                    str(j)+"_"+str(i)
                cv2.imwrite(os.path.join(self.config.SYN_DATASET_GEN.dataset_path,
                                         "img",
                                         file_name +".png"), image)
                
                # Save the synthetic mask as a numpy array
                np.save(os.path.join(self.config.SYN_DATASET_GEN.dataset_path,
                                     "mask",
                                     file_name+".npy"), obj_mask)
                
                # Write the synthetic metadata with the target condition in a csv file
                dict_data = {'filename':file_name, 'condition':target_condition}
                write_to_csv_from_dict(
                    dict_data = dict_data, 
                    csv_file_path=self.metadata_path, 
                    file_name=""
                )
                # with open(os.path.join(self.config.SYN_DATASET_GEN.dataset_path,
                #                        "metadata.csv"), 'a') as f:
                #     f.write(file_name+","+target_condition+"\n")
                    
                # Write the filename in a txt file
                with open(os.path.join(self.config.SYN_DATASET_GEN.dataset_path,
                                       "filenames.txt"), 'a') as f:
                    f.write(file_name+"\n")                
                
            
if __name__ == "__main__":
    config = omegaconf.OmegaConf.load('lang_data_synthesis/config.yaml')
    SEGMENTATION = config.SYN_DATASET_GEN.segmentation
    IMAGE_META_DATA = True
    VALIDATION = False
    dataset = WaymoDataset(config.IMAGE.WAYMO, 
                image_meta_data=IMAGE_META_DATA,
                segmentation=SEGMENTATION,
                        validation=VALIDATION)
    dataset_gen = SyntheticAVGenerator(source_dataset=dataset,
                                       config=config)
    dataset_gen.generate_synthetic_dataset()