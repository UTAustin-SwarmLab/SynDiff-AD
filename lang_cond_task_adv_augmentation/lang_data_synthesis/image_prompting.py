import argparse
import torch
import sys
import os
from typing import *

sys.path.append(os.path.join(os.getcwd(),'LLaVA/'))
from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from lang_data_synthesis.utils import write_to_csv_from_dict
from PIL import Image

import requests
from PIL import Image
from io import BytesIO

from lang_data_synthesis.utils import convert_pallette_segment
import omegaconf
from waymo_open_data import WaymoDataset

import pandas as pd
from tqdm import tqdm
import numpy as np
import functools
import torch.utils.data.dataloader as DataLoader
import tensorflow as tf

from bdd100k.data_loader import BDD100KDataset
from carla.data_loader import CARLADataset
from lang_data_synthesis.dataset import collate_fn as collator
from lang_data_synthesis.dataset import Dataset


import multiprocessing
from multiprocessing import Process, Queue

from argparse import ArgumentParser

class LLaVACaption:
    '''
    Class to generate captions for the Robotics Tasks.
    We are only captioning the training datasets
    '''

    def __init__(self, config:omegaconf, 
                 dataset:Dataset):
        self.config = config
        self.dataset:Dataset = dataset
    
        FILENAME = self.config.LLAVACAPTION.conditions_path 
        if not os.path.exists(FILENAME):
            self.conditions_metadata = None
        else:
            self.conditions_metadata = pd.read_csv(FILENAME)
        
        if isinstance(self.dataset, WaymoDataset):
            data_dict = {
                        "context_name":"content_name",
                        "context_frame":"context_frame",
                        "camera_id":"camera_id",
                        # "image_index":"image_index",
                        "caption":"caption"
            }
        elif isinstance(self.dataset, BDD100KDataset):
            data_dict = {
                        "file_name":"file_name",
                        "caption":"caption"
            }
        elif isinstance(self.dataset, CARLADataset):
            data_dict = {
                k:k for k in self.dataset.data_keys
            }
            
        if config.LLAVACAPTION.num_workers > 0:
            for j in range(self.config.LLAVACAPTION.num_workers):
                write_to_csv_from_dict(
                            dict_data=data_dict , 
                            csv_file_path= self.config.LLAVACAPTION.captions_path_multi.format(j),
                            file_name=""
                        )
        else:
             write_to_csv_from_dict(
                        dict_data=data_dict , 
                        csv_file_path= self.config.LLAVACAPTION.captions_path_single,
                        file_name=""
                        )
            
        collate_fn = functools.partial(
                collator, 
                segmentation=self.dataset.segmentation,
                image_meta_data=self.dataset.image_meta_data
            )
        
        self.dataloader = DataLoader.DataLoader(
            dataset=self.dataset,
            batch_size=self.config.LLAVACAPTION.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.LLAVACAPTION.batch_size
        )
        
    def load_image(self,
                index,
                batch_size: int = 1):
        if batch_size == 1:
            image, _, _, object_mask, image_data = self.dataset[index]
            prompt_tokens = self.dataset.get_text_description(object_mask)
            return image, prompt_tokens, image_data
        else:
            images = []
            prompt_tokens_list = []
            image_data_list = []
            for j in range(batch_size):
                image, _, _, object_mask, image_data = self.dataset[index*batch_size + j]
                images.append(image)
                prompt_tokens = self.dataset.get_text_description(object_mask)
                prompt_tokens_list.append(prompt_tokens)
                image_data_list.append(image_data)
            image = np.stack(images)

            return images, prompt_tokens_list, image_data_list

   
    @staticmethod   
    def get_caption(
            model: torch.nn.Module,
            model_name: str,
            image_processor: torch.nn.Module,
            tokenizer: torch.nn.Module,
            prompt: str,
            index: int,
            query: str,
            conv_mode: str,
            image=None,
            prompt_objects=None,
            image_data=None):

        # if image is None or prompt_objects is None or image_data is None:
        #     image, prompt_objects, image_data = self.load_image(
        #         index
        #     )
        qs = query.format(prompt_objects)

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        

        image_tensor = image_processor.preprocess(image, 
                                                return_tensors='pt')['pixel_values']\
                                                    .half().cuda()

        input_ids = tokenizer_image_token(prompt, 
                                        tokenizer, 
                                        IMAGE_TOKEN_INDEX,
                                        return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        return outputs, image_data
    
    @staticmethod
    def write_caption(
        conditions_metadata,
        dataset_type,
        image_data,
        file_path,
        caption=""
        ):
        if dataset_type == 'waymo':
            condition_data = conditions_metadata.loc[
                    (conditions_metadata['context_frame'] == image_data['context_frame']) &
                    (conditions_metadata['context_name'] == image_data['context_name']) &
                    (conditions_metadata['camera_id'] == image_data['camera_id'])
                ]['condition'].values.tolist()[0]
        elif dataset_type == 'bdd':
            condition_data = conditions_metadata.loc[
                    (conditions_metadata['file_name'] == image_data['file_name'])
            ]['condition'].values.tolist()[0]
        elif dataset_type == 'carla': 
            caption = " This image is taken during {} weather and day condition. ".format(image_data['condition']) + caption
        
        if dataset_type == 'waymo' or dataset_type == 'bdd':
            image_data['condition'] = condition_data
            weather = image_data['condition'].split(",")[0]
            time = image_data['condition'].split(",")[1]
            caption = " This image is taken during {} time of the day and features {} weather. ".format(time, weather) + caption
        
        if dataset_type == 'waymo':
            data_dict = {
                "context_name":image_data['context_name'],
                "context_frame":image_data['context_frame'],
                "camera_id":image_data['camera_id'],
                # "image_index":image_data['image_index'],
                "caption":caption
            }
        elif dataset_type == 'bdd':
            data_dict = {
                "file_name":image_data['file_name'],
                "caption":caption
            }
        elif dataset_type == 'carla':
            data_dict = image_data.copy()
            data_dict['caption'] = caption
        
        write_to_csv_from_dict(
                dict_data = data_dict, 
                csv_file_path= file_path,
                file_name=""
        )

    @staticmethod
    def worker_process(queue, 
                       model_path, 
                       model_base,
                       load_8bit, 
                       load_4bit,
                       gpu_id,
                       worker_id,
                       conditions_metadata,
                       get_text_description:Callable,
                       get_caption:Callable,
                       write_caption:Callable,
                       dataset_type = 'waymo',
                       
                       ):
        """
        Worker process that fetches a batch from the queue, processes it, and returns the result.
        """
        # Set the GPU for this worker
        torch.cuda.set_device(gpu_id)

        # Load the model in the worker
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, model_base, get_model_name_from_path(model_path),
            load_8bit=load_8bit, load_4bit=load_4bit,
        )

        while True:
            batch, config = queue.get()
            if batch is None:  # Poison pill means shutdown
                break

            with torch.no_grad():

            #for j in tqdm(range(self.config.LLAVACAPTION.NUM_IMAGES)):
                camera_images = batch[0]
                object_masks = batch[3]
                image_data_batch = batch[4]
                
                for j in range(len(camera_images)):
                    image = camera_images[j]
                    object_mask = object_masks[j]
                    image_data = image_data_batch[j]
                    prompt_tokens = get_text_description(object_mask)
                    caption, _ = get_caption(model, 
                                        model_name, 
                                        image_processor, 
                                        tokenizer, 
                                        config.LLAVACAPTION.prompt, 
                                        j, 
                                        config.LLAVACAPTION.prompt, 
                                        config.LLAVACAPTION.conv_mode,
                                        image=image,
                                        prompt_objects=prompt_tokens,
                                        image_data=image_data)
                    # keys = list(image_data.keys())
                    # values = list(image_data.values())
                    # column_names.append(tuple(keys + ['caption']))
                    # Modify the condition
                    write_caption(
                        conditions_metadata,
                        dataset_type,
                        image_data,
                        config.LLAVACAPTION.captions_path_multi.format(worker_id),
                        caption
                    )
                    
            #result_queue.put(True)
            
    def caption_dataset(self):
        # Model
        disable_torch_init()

        model_name = get_model_name_from_path(self.config.LLAVACAPTION.MODELPATH)
        if self.config.LLAVACAPTION.MODELBASE =='None':
            self.config.LLAVACAPTION.MODELBASE = None
        tokenizer, model, image_processor, context_len \
            = load_pretrained_model(self.config.LLAVACAPTION.MODELPATH, 
                                    self.config.LLAVACAPTION.MODELBASE,
                                    model_name,
                                    load_8bit=self.config.LLAVACAPTION.load_8bit,
                                    load_4bit=self.config.LLAVACAPTION.load_4bit,)

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.config.LLAVACAPTION.conv_mode is not None \
            and conv_mode != self.config.LLAVACAPTION.conv_mode:
            print('[WARNING] the auto inferred conversation mode is\
                {}, while `--conv-mode` is {}, using {}'.format(conv_mode, 
                                                        self.config.LLAVACAPTION.conv_mode, 
                                                        self.config.LLAVACAPTION.conv_mode))
        else:
            self.config.LLAVACAPTION.conv_mode = conv_mode

        captions = []
        column_names = []

        if self.config.LLAVACAPTION.NUM_IMAGES == 'ALL':
            self.config.LLAVACAPTION.NUM_IMAGES = len(dataset)
        elif type(self.config.LLAVACAPTION.NUM_IMAGES) == str:
            raise ValueError('NUM_IMAGES must be an integer or "ALL"')

        with torch.no_grad():
            for j, outputs in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            #for j in tqdm(range(self.config.LLAVACAPTION.NUM_IMAGES)):
                camera_images = outputs[0]
                object_masks = outputs[3]
                image_data_batch = outputs[4]
                
                for j in range(len(camera_images)):
                    image = camera_images[j]
                    object_mask = object_masks[j]
                    image_data = image_data_batch[j]
                    if isinstance(self.dataset, WaymoDataset):
                        prompt_tokens = WaymoDataset.get_text_description(object_mask,
                                                                          self.dataset.CLASSES)
                        dataset_type = 'waymo'
                    elif isinstance(self.dataset, BDD100KDataset):
                        prompt_tokens = BDD100KDataset.get_text_description(object_mask,
                                                                            self.dataset.CLASSES)
                        dataset_type = 'bdd'
                    elif isinstance(self.dataset, CARLADataset):
                        prompt_tokens = CARLADataset.get_text_description(object_mask, 
                                                                          self.dataset.CLASSES)
                        dataset_type = 'carla'
                    caption, _ = self.get_caption(model, 
                                        model_name, 
                                        image_processor, 
                                        tokenizer, 
                                        self.config.LLAVACAPTION.prompt, 
                                        j, 
                                        self.config.LLAVACAPTION.prompt, 
                                        self.config.LLAVACAPTION.conv_mode,
                                        image=image,
                                        prompt_objects=prompt_tokens,
                                        image_data=image_data)
                    # keys = list(image_data.keys())
                    # values = list(image_data.values())
                    # column_names.append(tuple(keys + ['caption']))
                    # Modify the condition
                    
                    # condition_data = self.conditions_metadata.loc[
                    #     (self.conditions_metadata['context_frame'] == image_data['context_frame']) &
                    #     (self.conditions_metadata['context_name'] == image_data['context_name']) &
                    #     (self.conditions_metadata['camera_id'] == image_data['camera_id'])
                    # ]['condition'].values.tolist()[0]
                    LLaVACaption.write_caption(
                        self.conditions_metadata,
                        dataset_type,
                        image_data,
                        config.LLAVACAPTION.captions_path_single,
                        caption
                    )

            # captions.append([tuple(values + [caption])])
        # column_names = set(column_names)
        # captions_df = pd.DataFrame(captions, columns=list(column_names))
        # return captions_df
    def caption_dataset_parallel(self, num_workers=4):
        """
        Function to caption the dataset using multiple processes with a model loaded in each.
        """
        # Create queues for data and results
        queue = Queue(maxsize=num_workers*3)
        # result_queue = Queue(maxsize=num_workers*3)
        if self.config.LLAVACAPTION.MODELBASE =='None':
            self.config.LLAVACAPTION.MODELBASE = None
        
        model_name = get_model_name_from_path(self.config.LLAVACAPTION.MODELPATH)
        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.config.LLAVACAPTION.conv_mode is not None \
            and conv_mode != self.config.LLAVACAPTION.conv_mode:
            print('[WARNING] the auto inferred conversation mode is\
                {}, while `--conv-mode` is {}, using {}'.format(conv_mode, 
                                                        self.config.LLAVACAPTION.conv_mode, 
                                                        self.config.LLAVACAPTION.conv_mode))
        else:
            self.config.LLAVACAPTION.conv_mode = conv_mode
        
        if isinstance(self.dataset, WaymoDataset):
            get_text_description = functools.partial(WaymoDataset.get_text_description, 
                                                 CLASSES=self.dataset.CLASSES)
            dataset_type = 'waymo'
        elif isinstance(self.dataset, BDD100KDataset):
            get_text_description = functools.partial(BDD100KDataset.get_text_description, 
                                                 CLASSES=self.dataset.CLASSES)
            dataset_type = 'bdd'
        elif isinstance(self.dataset, CARLADataset):
            get_text_description = functools.partial(
                CARLADataset.get_text_description,
                 CLASSES=self.dataset.CLASSES)
            dataset_type = 'carla'
        
        workers = [Process(target=LLaVACaption.worker_process, 
                           args=(queue,
                                 self.config.LLAVACAPTION.MODELPATH,
                                 self.config.LLAVACAPTION.MODELBASE,
                                 self.config.LLAVACAPTION.load_8bit,
                                 self.config.LLAVACAPTION.load_4bit,
                                 int(i / self.config.LLAVACAPTION.model_per_gpu),
                                 i,
                                 self.conditions_metadata,
                                 get_text_description,
                                 LLaVACaption.get_caption,
                                 LLaVACaption.write_caption,
                                 dataset_type
                                 ))
                   for i in range(num_workers)]

        for w in workers:
            w.start()

        # Feed batches to the queue
        for j, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            queue.put((batch,self.config))

        # Add poison pills to stop the workers
        for _ in range(num_workers):
            queue.put((None,None))

        # Collect results
        # all_captions = []
        # for _ in range(len(self.dataloader)):
        #     all_captions.extend(result_queue.get())

        # Wait for all worker processes to finish
        for w in workers:
            w.join()

        # return all_captions

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

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    config_file_name = 'lang_data_synthesis/{}_config.yaml'.format(args.experiment)
    config = omegaconf.OmegaConf.load(config_file_name)
    
    tf.config.set_visible_devices([], 'GPU')
    
    if args.experiment == 'waymo':
        dataset = WaymoDataset(config.IMAGE.WAYMO, image_meta_data=True)
    elif args.experiment =='bdd':
        dataset = BDD100KDataset(config.IMAGE.BDD, image_meta_data=True)
    elif args.experiment == 'carla':
        dataset = CARLADataset(config, image_meta_data=True)
    else:
        raise NotImplementedError
    
    captioner = LLaVACaption(config, dataset=dataset)
    if config.LLAVACAPTION.num_workers > 0:
        multiprocessing.set_start_method('spawn', force=True)
        captions = captioner.caption_dataset_parallel(config.LLAVACAPTION.num_workers)
    else:
        captions = captioner.caption_dataset()

    # Create a parquet file that stores captions of all the images in the dataset
    #prompt_folder = "../waymo_data/training/"

    # save captions 
    # print('Writing Captions')
    # captions.to_csv(os.path.join(prompt_folder,'waymo_captions.csv'), index=False)