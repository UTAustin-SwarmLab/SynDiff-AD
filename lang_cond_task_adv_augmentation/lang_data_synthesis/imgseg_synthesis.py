'''
Simple tool for generating images with control net conditioned on the segmenation masks.
We will use gradio to store the resulting image and then use it to perform predictions.

'''

import sys
import os
sys.path.append(os.path.join(os.getcwd(),'ControlNet/'))

import ControlNet.config
from ControlNet.cldm.hack import disable_verbosity, enable_sliced_attention
from waymo_open_data import WaymoDataset
from copy import deepcopy
disable_verbosity()

if ControlNet.config.save_memory:
    enable_sliced_attention()


import omegaconf
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random


from pytorch_lightning import seed_everything
from ControlNet.annotator.util import resize_image, HWC3
# from ControlNet.annotator.uniformer import UniformerDetector
# from ControlNet.annotator.oneformer import OneformerCOCODetector, OneformerADE20kDetector
from ControlNet.cldm.model import create_model, load_state_dict
from ControlNet.cldm.ddim_hacked import DDIMSampler
import pickle as pkl
from ADE20K.utils import utils_ade20k
import functools
import matplotlib.pyplot as plt
from typing import *
from lang_data_synthesis.utils import convert_pallette_segment
# from ControlNet.annotator.uniformer.mmseg.datasets import ADE20KDataset
from copy import copy
from bdd100k.data_loader import BDD100KDataset
from argparse import ArgumentParser
from carla.data_loader import CARLADataset
import pandas as pd
def gradio_viz(
        title:str,
        design_list:List[Dict]=None,
        synthesizer=None, 
        server_name=None, 
        input_image=None, 
        seg_mask = None, 
        server_port=None,
        prompt_tokens=None):
    '''
    Function for visualizing the segmentation maps and the resulting images.
    Args:
        design_list: List of objects to add with design components
             dictionary containing the design parameters
    '''
    # TODO: Modularisation based on the data dictionary
    block = gr.Blocks().queue()

    if synthesizer is None or synthesizer.model is None or synthesizer.ddim_sampler is None:
        raise ValueError("Please initialize the model first")
    
    with block:
        with gr.Row():
            gr.Markdown("{}".format(title))
        with gr.Row():
            with gr.Column():
                if input_image is not None:
                    input_image = gr.Image(label='Input Image', value=input_image,
                                        type='numpy')
                else:
                    input_image = gr.Image(label='Input Image', source='upload',
                                        type='numpy')
                if seg_mask is not None:
                    seg_mask_ = gr.Image(label='Segmentation Mask', value=seg_mask[0],
                                        type='numpy')
                
                det = gr.Radio(choices=["Seg_OFADE20K", "Seg_OFCOCO", "Seg_UFADE20K", "None"],
                                type="value", value="Seg_OFADE20K", label="Preprocessor")
                prompt = gr.Textbox(label="Prompt")
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    num_samples = gr.Slider(label="Images", minimum=1,
                                             maximum=12, value=1, step=1)
                    image_resolution = gr.Slider(label="Image Resolution", 
                                                 minimum=256, maximum=1024, value=512, step=64)
                    strength = gr.Slider(label="Control Strength", minimum=0.0,
                                          maximum=2.0, value=1.0, step=0.01)
                    init_noise = gr.Slider(label="Initialisation Noise", minimum=0.0,
                                          maximum=1.0, value=0.2, step=0.01)
                    guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                    apply_image_cond = gr.Checkbox(label='Apply Image Condition', value=False)
                    detect_resolution = gr.Slider(label="Segmentation Resolution", 
                                                  minimum=128, maximum=2048, 
                                                  value=512, step=1)
                    ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100,
                                            value=20, step=1)
                    scale = gr.Slider(label="Guidance Scale", minimum=0.1, 
                                      maximum=30.0, value=9.0, step=0.1)
                    seed = gr.Slider(label="Seed", minimum=-1, 
                                     maximum=2147483647, step=1, randomize=True)
                    eta = gr.Number(label="eta (DDIM)", value=0.0)
                    a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed, same textures, maintain all semantics')
                    n_prompt = gr.Textbox(label="Negative Prompt",
                                        value='disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w, cropped, worst quality, low quality')
            with gr.Column():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
        ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
               detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, det,
               apply_image_cond, init_noise]
        run_button.click(fn=functools.partial(synthesizer.process,
                                                seg_mask=seg_mask, prompt_tokens=prompt_tokens),
                            inputs=ips, outputs=[result_gallery])
               
    if server_port is None:
        block.launch(server_name=server_name)
    else:
        #block.launch(server_name=server_name, server_port=server_port)
        block.launch(share=True)


def get_ade20k(config):
    index_file = 'index_ade20k.pkl'
    dataset_path = os.path.join(os.getcwd(), config.IMAGE.ADE20K.DATASET_PATH)
    with open('{}/{}'.format(dataset_path, index_file), 'rb') as f:
        index_ade20k = pkl.load(f)
    
    file_name = index_ade20k['filename'][config.IMAGE.ADE20K.image_id]
    num_obj = index_ade20k['objectPresence'][:, config.IMAGE.ADE20K.image_id].sum()
    num_parts = index_ade20k['objectIsPart'][:, config.IMAGE.ADE20K.image_id].sum()
    count_obj = index_ade20k['objectPresence'][:, config.IMAGE.ADE20K.image_id].max()
    obj_id = np.where(index_ade20k['objectPresence'][:, config.IMAGE.ADE20K.image_id]\
                       == count_obj)[0][0]
    obj_name = index_ade20k['objectnames'][obj_id]
    full_file_name = '{}/{}'.format(index_ade20k['folder'][config.IMAGE.ADE20K.image_id],
                                     index_ade20k['filename'][config.IMAGE.ADE20K.image_id])

    print("The image at index {} is {}".format(config.IMAGE.ADE20K.image_id, file_name))
    print("It is located at {}".format(full_file_name))
    print("It happens in a {}".format(index_ade20k['scene'][config.IMAGE.ADE20K.image_id]))
    print("It has {} objects, of which {} are parts".format(num_obj, num_parts))
    print("The most common object is object {} ({}), which \
          appears {} times".format(obj_name, obj_id, count_obj))

    info = utils_ade20k.loadAde20K('{}/{}'.format(
        config.IMAGE.ADE20K.DATASET_PATH, 
        full_file_name
    ))
    img = cv2.imread(info['img_name'])[:,:,::-1]
    seg = cv2.imread(info['segm_name'])[:,:,::-1]
    object_mask = info['class_mask']
    seg_mask = seg.copy()
    
    # Map the object instance mask to that compatible with the color pallette
    dataset_metadata = ADE20KDataset()
    metadata = {}
    metadata['object_classes'] = dataset_metadata.CLASSES
    metadata['pallete'] = dataset_metadata.PALETTE
    extra_mapping = dict()
    extra_mapping['central reservation'] = 'sidewalk'

    # Somehow the indexes are shifted by 1: correcting this
    object_mask -=1 
    object_mask[object_mask == -1] = 0

    seg_mask_metadata = {}
    seg_mask_metadata['object_classes'] = copy(index_ade20k['objectnames'])
    seg_mask, object_keys = convert_pallette_segment(metadata=metadata,
                                        obj_mask=object_mask.copy(),
                                        seg_mask_metadata=seg_mask_metadata,
                                        extra_mapping=extra_mapping)
    
    return img, seg_mask, object_keys


def get_waymo(config):
    SEGMENTATION = True
    IMAGE_META_DATA = True
    VALIDATION = False
    dataset = WaymoDataset(config.IMAGE.WAYMO, 
                image_meta_data=IMAGE_META_DATA,
                segmentation=SEGMENTATION,
                        validation=VALIDATION)

    if config.SYN_DATASET_GEN.use_llava_prompt:
        llavafilename = os.path.join(config.SYN_DATASET_GEN.llava_prompt_path)
        prompt_df = pd.read_csv(llavafilename)
        metadata_conditions = pd.read_csv(config.ROBUSTIFICATION.train_file_path)
    # Obtain the image
    #np.random.seed(200)
    idx = np.random.randint(0, len(dataset))
    camera_images, semantic_mask_rgb, instance_masks, object_masks, img_data = dataset[idx]
    
    # Obtain the segmentation mask
    prompt_tokens = {}
    prompt_tokens['a_prompt'] = "Must contain " + WaymoDataset.get_text_description(object_masks, dataset.CLASSES)
    prompt_tokens['n_prompt'] = "Must not contain " +  WaymoDataset.get_text_description(object_masks, dataset.CLASSES)
    prompt = None
    source_condition = None
    if prompt_df is not None:
        if 'context_name' in img_data.keys():
            prompt = prompt_df.loc[
                        (img_data['context_name'] == prompt_df['context_name']) &
                        (img_data['context_frame'] == prompt_df['context_frame']) &
                        (img_data['camera_id'] == prompt_df['camera_id'])
            ]['caption'].values[0]
            source_condition = metadata_conditions.loc[
                    (img_data['context_name'] == metadata_conditions['context_name']) &
                    (img_data['context_frame'] == metadata_conditions['context_frame']) &
                    (img_data['camera_id'] == metadata_conditions['camera_id'])
                ]['condition'].values[0]
                    
        
    semantic_mapped_rgb_mask = dataset.get_mapped_semantic_mask(object_masks)
    invalid_mask = dataset.get_unmapped_mask(object_masks)
    return camera_images, semantic_mapped_rgb_mask, prompt_tokens, invalid_mask, prompt, source_condition

def get_bdd(config):
    SEGMENTATION = True
    IMAGE_META_DATA = True
    VALIDATION = False
    
    dataset = BDD100KDataset(config.IMAGE.BDD, 
                image_meta_data=IMAGE_META_DATA,
                segmentation=SEGMENTATION,
                        validation=VALIDATION)

    prompt_df = None
    if config.SYN_DATASET_GEN.use_llava_prompt:
        llavafilename = os.path.join(config.SYN_DATASET_GEN.llava_prompt_path)
        prompt_df = pd.read_csv(llavafilename)
        metadata_conditions = pd.read_csv(config.ROBUSTIFICATION.train_file_path)
        
    # Obtain the image
    #np.random.seed(200)
    idx = np.random.randint(0, len(dataset))
    camera_images, semantic_mask_rgb, instance_masks, object_masks, img_data = dataset[idx]
    
    # Obtain the segmentation mask
    prompt_tokens = {}
    prompt_tokens['a_prompt'] = "Must contain " + BDD100KDataset.get_text_description(object_masks, dataset.CLASSES)
    prompt_tokens['n_prompt'] = "Must not contain " +  BDD100KDataset.get_text_description(object_masks, dataset.CLASSES)
    
    semantic_mapped_rgb_mask = dataset.get_mapped_semantic_mask(object_masks)
    invalid_mask = dataset.get_unmapped_mask(object_masks)
    prompt = None
    source_condition = None
    if prompt_df is not None:
        prompt = prompt_df.loc[
                    (img_data['file_name'] == prompt_df['file_name'])
                ]['caption'].values[0]
        source_condition = metadata_conditions.loc[
                    (img_data['file_name'] == metadata_conditions['file_name'])
                ]['condition'].values[0]
    return camera_images, semantic_mapped_rgb_mask, prompt_tokens, invalid_mask, prompt, source_condition

def get_carla(config):
    SEGMENTATION = True
    IMAGE_META_DATA = True
    VALIDATION = False
    
    dataset = CARLADataset(config, 
                image_meta_data=IMAGE_META_DATA,
                segmentation=SEGMENTATION,
                        validation=VALIDATION)

    prompt_df = None
    if config.SYN_DATASET_GEN.use_llava_prompt:
        llavafilename = os.path.join(config.SYN_DATASET_GEN.llava_prompt_path)
        prompt_df = pd.read_csv(llavafilename)

        
    # Obtain the image
    #np.random.seed(200)
    idx = np.random.randint(0, len(dataset))
    camera_images, semantic_mask_rgb, instance_masks, object_masks, img_data = dataset[idx]
    
    # Obtain the segmentation mask
    prompt_tokens = {}
    prompt_tokens['a_prompt'] = "Must contain " + CARLADataset.get_text_description(object_masks, dataset.CLASSES)
    prompt_tokens['n_prompt'] = "Must not contain " +  CARLADataset.get_text_description(object_masks, dataset.CLASSES)
    
    semantic_mapped_rgb_mask = dataset.get_mapped_semantic_mask(object_masks)
    invalid_mask = dataset.get_unmapped_mask(object_masks)
    prompt = None
    source_condition = None
    if prompt_df is not None:
        prompt = prompt_df.loc[
                    (img_data['route'] == prompt_df['route'])&
                    (img_data['file_name']==prompt_df['file_name'])&
                    (img_data['mask_path'] == prompt_df['mask_path'])&
                    (img_data['synthetic'] == prompt_df['synthetic'])&
                    (img_data['condition'] == prompt_df['condition'])
                ]['caption'].values[0]
        source_condition = img_data['condition']
    return camera_images, semantic_mapped_rgb_mask, prompt_tokens, invalid_mask, prompt, source_condition

class ImageSynthesis:
    
    def __init__(self, config) -> None:
        self.config = config
        self.initialize_model()
        
    def initialize_model(
        self  
    ):
        
        if self.config.SYN_DATASET_GEN.use_finetuned:
            self.config.control_net.MODEL_PATH = os.path.join(os.getcwd(),
                                                    self.config.control_net.FT_MODEL_PATH)
        else:
            self.config.control_net.MODEL_PATH = os.path.join(os.getcwd(),
                                                    self.config.control_net.MODEL_PATH)
        self.config.control_net.CONFIG_PATH = os.path.join(os.getcwd(), 
                                                    self.config.control_net.CONFIG_PATH)
        self.config.control_net.SD_CHECKPOINT = os.path.join(os.getcwd(),
                                                        self.config.control_net.SD_CHECKPOINT)
        self.model = create_model(self.config.control_net.CONFIG_PATH).cpu()
        self.model.load_state_dict(load_state_dict(self.config.control_net.SD_CHECKPOINT, 
                                            location='cuda'), 
                                            strict=False)
        self.model.load_state_dict(load_state_dict(self.config.control_net.MODEL_PATH,
                                            location='cuda'),
                                            strict=False)
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)


    def process(
        self, 
        input_image,
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        image_resolution,
        detect_resolution, 
        ddim_steps,
        guess_mode,
        strength,
        scale,
        seed,
        eta,
        det,
        apply_image_cond=False,
        init_noise=0.2,
        seg_mask=None,
        prompt_tokens=None,
        finetuned_control = False
        ):
        '''
        prompt_tokens : Dict of additional prompt tokens
        '''

        # if det == 'Seg_OFCOCO':
        #     preprocessor = OneformerCOCODetector()
        # if det == 'Seg_OFADE20K':
        #     preprocessor = OneformerADE20kDetector()
        # if det == 'Seg_UFADE20K':
        #     preprocessor = UniformerDetector()
                            
        with torch.no_grad():
            input_image = HWC3(input_image)
            if seg_mask is None and det is None:
                detected_map = input_image.copy()
            elif seg_mask is None:
                # detected_map = preprocessor(resize_image(input_image, detect_resolution))
                uniformer_segmap = detected_map.copy()
            else:
                # uniformer_segmap = preprocessor(resize_image(input_image, detect_resolution))
                detected_map = seg_mask[0].copy()

            if finetuned_control:
                img = cv2.resize(input_image, (image_resolution, image_resolution),
                                 interpolation=cv2.INTER_AREA)
            else:
                img = resize_image(input_image, image_resolution)
                
            H, W, C = img.shape
            
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if ControlNet.config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            # Add more prompt keywords based on the semantics present
            
            
            if prompt_tokens is not None and prompt_tokens['a_prompt'] is not None:
                a_prompt += ("," + prompt_tokens['a_prompt'])
            if prompt_tokens is not None and prompt_tokens['n_prompt'] is not None:
                n_prompt += ("," + prompt_tokens['n_prompt'])

            if isinstance(prompt, str):
                cond = {"c_concat": [control], "c_crossattn": 
                        [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            elif isinstance(prompt, list):
                if len(prompt) > num_samples:
                    prompt = prompt[:num_samples]
                elif len(prompt) < num_samples:
                    prompt = prompt + [prompt[-1]] * (num_samples - len(prompt))
                
                cond = {"c_concat": [control], "c_crossattn": 
                        [self.model.get_learned_conditioning([prompt[j] + ',' +a_prompt for j in range(num_samples)])]}
                
            un_cond = {"c_concat": None if guess_mode else [control],
                        "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if ControlNet.config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) \
                                    for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            
            img_cond = torch.tensor(img).float().cuda().permute((2,0,1)) / 255.0
            encoder_posterior = self.model.encode_first_stage(img_cond.unsqueeze(0).repeat(num_samples, 1, 1, 1))            
            z = self.model.get_first_stage_encoding(encoder_posterior).detach()
            x_T = init_noise*torch.randn_like(z) + z * (1 - init_noise)
            
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        x_T=x_T,
                                                        unconditional_conditioning=un_cond)

            if ControlNet.config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5)\
                .cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
            seg_results = []
            # for result in results:
            #     seg_results.append(preprocessor(resize_image(result, detect_resolution)))
        return [detected_map] + results
        #return [uniformer_segmap] + [detected_map] + results #+ seg_results
        
    def run_model(
        self,
        input_image: np.ndarray,
        seg_mask:np.ndarray =None,
        prompt:str = None,
        prompt_tokens:dict = None
    ):
        
        if input_image is None:
            raise ValueError("Please provide the input image")
        
        if prompt is None:
            prompt = 'Rainy weather condition during nightime'
        
        # Obtain from config
        a_prompt = self.config.SYNTHESIS_PARAMS.A_PROMPT
        n_prompt = self.config.SYNTHESIS_PARAMS.N_PROMPT
        guess_mode = self.config.SYNTHESIS_PARAMS.GUESS
        detect_resolution = self.config.SYNTHESIS_PARAMS.DETECT_RESOLUTION
        image_resolution = self.config.SYNTHESIS_PARAMS.IMAGE_RESOLUTION
        ddim_steps = self.config.SYNTHESIS_PARAMS.DDIMSTEPS
        scale = self.config.SYNTHESIS_PARAMS.SCALE
        seed = self.config.SYNTHESIS_PARAMS.SEED
        eta = self.config.SYNTHESIS_PARAMS.ETA
        num_samples = self.config.SYNTHESIS_PARAMS.NUMSAMPLES
        strength = self.config.SYNTHESIS_PARAMS.CONTROL
        init_noise = self.config.SYNTHESIS_PARAMS.INIT_NOISE
        annotator = 'Seg_OFADE20K'

        outputs = self.process(input_image, prompt, a_prompt, n_prompt,
                    num_samples, image_resolution, detect_resolution,
                    ddim_steps, guess_mode, strength, scale, seed, eta,
                    init_noise=init_noise, det=annotator,
                    seg_mask=seg_mask, prompt_tokens=prompt_tokens,
                    finetuned_control=self.config.SYN_DATASET_GEN.use_finetuned)

        return outputs

def post_process(invalid_mask, syn_img, gt_img):
    '''
    Post processing function for the synthetic image
    '''
    syn_img = cv2.resize(syn_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    syn_img[invalid_mask.squeeze()] = gt_img[invalid_mask.squeeze()]
    return syn_img

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
        default=0,
        help='Offset seed for random number generators')
    return  parser.parse_args()


def output_single(config):
    synthesizer = ImageSynthesis(config)
   
    #img, seg_mask, object_keys = get_ade20k(config)
    for k in range(20):
        seed = np.random.seed()
        np.random.seed(seed)
        if args.experiment == 'waymo':
            img, seg_mask, object_keys, mask, prompt, weather = get_waymo(config)
        elif args.experiment == 'bdd':
            img, seg_mask, object_keys, mask, prompt, weather = get_bdd(config)
        elif args.experiment == 'carla':
            img, seg_mask, object_keys, mask, prompt, weather = get_carla(config)
            
        if isinstance(object_keys, list):
            prompt_tokens = {'a_prompt': '', 'n_prompt': ''}
            for key in object_keys:
                if key != '-':
                    prompt_tokens['a_prompt'] += 'same {}, '.format(key)
                    prompt_tokens['n_prompt'] += 'missing {}, '.format(key)
        else:
            prompt_tokens = object_keys
            
        if config.GRADIO:
            gradio_viz(title='control net test', 
                    synthesizer=synthesizer,
                    input_image=img.copy(), seg_mask = [seg_mask.copy()], 
                    server_name='hg22723@swarmcluster1.ece.utexas.edu', server_port=8090,
                    prompt_tokens=prompt_tokens)
        else:
            if prompt is None:
                prompt = 'Rainy weather condition during nightime'
            else:
                if args.experiment == 'carla':
                    prompt = prompt.replace(weather, 'RainyNight')
                else:
                    target_condition = "Rainy, Night"
                    source_condition = weather
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
                            
            outputs = synthesizer.run_model(img.copy(), 
                                seg_mask=[seg_mask.copy()],
                                prompt = prompt,
                                prompt_tokens=prompt_tokens)
                    
            results = outputs[1:]
            for j,syn_img in enumerate(results):            
                syn_img = cv2.resize(syn_img, (img.shape[1],
                                           img.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
                outputs[j+1] = syn_img
            
            #     outputs[j+1] = post_process(mask, syn_img, img)
            save_path = os.path.join(os.getcwd(), config.SAVE.PATH)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.savefig(os.path.join(save_path, 'input_{}.png'.format(k)))
            plt.close()
            for i, output in enumerate(outputs):
                plt.figure(figsize=(10, 10))
                plt.imshow(output)
                plt.savefig(os.path.join(save_path, 'output_{}_{}.png'.format(k,i+1)))
                plt.close()
                
def output_styles(args, config):
    #img, seg_mask, object_keys = get_ade20k(config)
    # Choose images 
    
    synthesizer = ImageSynthesis(config)
    #img, seg_mask, object_keys = get_ade20k(config)
    for k in range(20):
        np.random.seed(k*10+8)
        if args.experiment == 'waymo':
            img, seg_mask, object_keys, mask, prompt, weather = get_waymo(config)
            # Maps for abbr
            ALL_CONDITIONS = {"Rainy, Night":"rnn", 
                              "Rainy, Day":"rd", 
                              "Rainy, Dawn/Dusk":"rdd",
                              "Clear, Night":"cn",
                              "Clear, Day":"cd",
                              "Clear, Dawn/Dusk":"cdd",
                              "Cloudy, Night":"cln", 
                              "Cloudy, Day":"cld",
                              "Cloudy, Dawn/Dusk":"cldd"}
        elif args.experiment == 'bdd':
            img, seg_mask, object_keys, mask, prompt, weather = get_bdd(config)
            ALL_CONDITIONS = {"Rainy, Night":"rnn", 
                              "Rainy, Day":"rd", 
                              "Rainy, Dawn/Dusk":"rdd",
                              "Clear, Night":"cn",
                              "Clear, Day":"cd",
                              "Clear, Dawn/Dusk":"cdd",
                              "Cloudy, Night":"cln", 
                              "Cloudy, Day":"cld",
                              "Cloudy, Dawn/Dusk":"cldd"}
        elif args.experiment == 'carla':
            img, seg_mask, object_keys, mask, prompt, weather = get_carla(config)
            ALL_CONDITIONS = {"RainNight":"rnn", 
                              "RainMorning":"rd", 
                              "RainTwilight":"rt",
                              "ClearNight":"cn",
                              "ClearMorning":"cd",
                              "ClearTwilight":"ct",
                              "CloudyNight":"cln", 
                              "CloudyMorning":"cld",
                              "CloudyTwilight":"clt"}
        
        #     outputs[j+1] = post_process(mask, syn_img, img)
        save_path = os.path.join(os.getcwd(), config.SAVE.PATH+"/{}".format(args.experiment))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.savefig(os.path.join(save_path, 'input_{}_{}.png'.format(k, ALL_CONDITIONS[weather])))
        plt.close()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(seg_mask)
        plt.savefig(os.path.join(save_path, 'inputmask_{}_{}.png'.format(k, ALL_CONDITIONS[weather])))
        plt.close()
        
        # Obtain all the remaining conditions to be plotted
        remaining_conditions = list(ALL_CONDITIONS.keys())
        remaining_conditions.remove(weather)
        
        source_condition = weather
        for target_condition in remaining_conditions:
            # Obtaining the images
            img_prompt = deepcopy(prompt)
            if isinstance(object_keys, list):
                prompt_tokens = {'a_prompt': '', 'n_prompt': ''}
                for key in object_keys:
                    if key != '-':
                        prompt_tokens['a_prompt'] += 'same {}, '.format(key)
                        prompt_tokens['n_prompt'] += 'missing {}, '.format(key)
            else:
                prompt_tokens = object_keys
                
            if config.GRADIO:
                gradio_viz(title='control net test', 
                        synthesizer=synthesizer,
                        input_image=img.copy(), seg_mask = [seg_mask.copy()], 
                        server_name='hg22723@swarmcluster1.ece.utexas.edu', server_port=8090,
                        prompt_tokens=prompt_tokens)
            else:
                if img_prompt is None:
                    img_prompt = '{} weather condition during {} time'.format(target_condition.split(',')[0],
                                                                          target_condition.split(',')[1])
                else:
                    if args.experiment == 'carla':
                        img_prompt = prompt.replace(weather, target_condition)
                    else:

                        weather, day = target_condition.split(',') 
                        source_weather, source_day = source_condition.split(',')
                        
                        img_prompt = img_prompt.replace(source_weather, weather)
                        img_prompt = img_prompt.replace(source_day, day)
                        # remove all the words associated with the source condition

                        img_prompt = img_prompt.replace(source_day.lower(), day)   
                        
                        if 'y' in source_weather:
                            img_prompt = img_prompt.replace(' '+source_weather[:-1].lower(), ' '+weather)
                            if 'y' in weather:
                                img_prompt = img_prompt.replace(source_weather[:-1], weather[:-1])
                            else:
                                img_prompt = img_prompt.replace(source_weather[:-1], weather)
                        else:
                            img_prompt = img_prompt.replace(source_weather.lower(), weather)
                                
                outputs = synthesizer.run_model(img.copy(), 
                                    seg_mask=[seg_mask.copy()],
                                    prompt = img_prompt,
                                    prompt_tokens=prompt_tokens)
                        
                results = outputs[1:]
                for j,syn_img in enumerate(results):            
                    syn_img = cv2.resize(syn_img, (img.shape[1],
                                            img.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
                    outputs[j+1] = syn_img
                

                for i, output in enumerate(outputs):
                    plt.figure(figsize=(10, 10))
                    plt.imshow(output)
                    plt.savefig(os.path.join(save_path, 'output_{}_{}_{}.png'.format(k,i+1,
                                                                                     ALL_CONDITIONS[target_condition])))
                    plt.close()

if __name__ == "__main__":
    
    args = parse_args()
    
    config_file_name = 'lang_data_synthesis/{}_config.yaml'.format(args.experiment)
    config = omegaconf.OmegaConf.load(config_file_name)
    
    #output_single(config)
    
    output_styles(args, config)
                              
# TODO: Modularization of this function is pending
# DESIGN_DICT =  [{'type': 'image', 
#                'title': 'Input Image', 
#                'key': 'input_image', 
#                'source': 'upload', 
#                'type': 'numpy'},
#                 {'type': 'textbox',
#                  'title': 'Prompt',
#                  'key': 'prompt'},