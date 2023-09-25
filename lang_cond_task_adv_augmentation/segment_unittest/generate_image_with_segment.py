'''
Simple tool for generating images with control net conditioned on the segmenation masks.
We will use gradio to store the resulting image and then use it to perform predictions.

'''

import sys
import os
sys.path.append(os.path.join(os.getcwd(),'ControlNet/'))

import ControlNet.config
from ControlNet.cldm.hack import disable_verbosity, enable_sliced_attention


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
from ControlNet.annotator.uniformer import UniformerDetector
from ControlNet.annotator.oneformer import OneformerCOCODetector, OneformerADE20kDetector
from ControlNet.cldm.model import create_model, load_state_dict
from ControlNet.cldm.ddim_hacked import DDIMSampler
import pickle as pkl
from ADE20K.utils import utils_ade20k
import functools
import matplotlib.pyplot as plt
from typing import *
from segment_unittest.utils import convert_pallette_segment
from ControlNet.annotator.uniformer.mmseg.datasets import ADE20KDataset
from copy import copy

def gradio_viz(
        title:str,
        design_list:List[Dict]=None,
        model=None, 
        ddim_sampler=None, 
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

    if model is None or ddim_sampler is None:
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
                    guess_mode = gr.Checkbox(label='Guess Mode', value=False)
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
               detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, det]
        run_button.click(fn=functools.partial(process,
                                                model=model, ddim_sampler=ddim_sampler,
                                                seg_mask=seg_mask, prompt_tokens=prompt_tokens),
                            inputs=ips, outputs=[result_gallery])
               
    if server_port is None:
        block.launch(server_name=server_name)
    else:
        #block.launch(server_name=server_name, server_port=server_port)
        block.launch(share=True)

def run_model(
        input_image,
        model=None, 
        ddim_sampler=None,
        seg_mask=None
        ):
    
    if input_image is None:
        raise ValueError("Please provide the input image")
    
    prompt = 'Night'
    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'extra objects,fewer objects, cropped, clouds, daylight, worst quality, low quality'
    strength = 0.1
    num_samples = 2
    guess_mode = False
    detect_resolution = 512
    image_resolution = 512
    ddim_steps = 20
    scale = 9.0
    seed = 2000
    eta = 30.0
    guess_mode = True
    annotator = 'Seg_OFADE20K'

    outputs = process(input_image, prompt, a_prompt, n_prompt,
                num_samples, image_resolution, detect_resolution,
                ddim_steps, guess_mode, strength, scale, seed, eta,
                det=annotator, model=model, ddim_sampler=ddim_sampler,
                  seg_mask=seg_mask)
    
    save_path = os.path.join(os.getcwd(), config.SAVE.PATH)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(input_image)
    plt.savefig(os.path.join(save_path, 'input.png'))
    plt.close()
    for i, output in enumerate(outputs):
        plt.figure(figsize=(10, 10))
        plt.imshow(output)
        plt.savefig(os.path.join(save_path, 'output_{}.png'.format(i+1)))
        plt.close()


def initialize_model(config):
    config.control_net.MODEL_PATH = os.path.join(os.getcwd(),
                                                 config.control_net.MODEL_PATH)
    config.control_net.CONFIG_PATH = os.path.join(os.getcwd(), 
                                                  config.control_net.CONFIG_PATH)
    config.control_net.SD_CHECKPOINT = os.path.join(os.getcwd(),
                                                     config.control_net.SD_CHECKPOINT)
    model = create_model(config.control_net.CONFIG_PATH).cpu()
    model.load_state_dict(load_state_dict(config.control_net.SD_CHECKPOINT, 
                                          location='cuda'), 
                                          strict=False)
    model.load_state_dict(load_state_dict(config.control_net.MODEL_PATH,
                                           location='cuda'),
                                           strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    return model, ddim_sampler

def process(input_image, prompt, a_prompt, n_prompt,
             num_samples, image_resolution, detect_resolution, 
             ddim_steps, guess_mode, strength, scale, seed, eta, det,
             apply_uniformer=None, model=None, ddim_sampler=None, seg_mask=None,
             prompt_tokens=None):
    '''
    prompt_tokens : Dict of additional prompt tokens
    '''

    if det == 'Seg_OFCOCO':
        preprocessor = OneformerCOCODetector()
    if det == 'Seg_OFADE20K':
        preprocessor = OneformerADE20kDetector()
    if det == 'Seg_UFADE20K':
        preprocessor = UniformerDetector()
                          
    with torch.no_grad():
        input_image = HWC3(input_image)
        if seg_mask is None and det is None:
            detected_map = input_image.copy()
        elif seg_mask is None:
            detected_map = preprocessor(resize_image(input_image, detect_resolution))
            uniformer_segmap = detected_map.copy()
        else:
            uniformer_segmap = preprocessor(resize_image(input_image, detect_resolution))
            detected_map = seg_mask[0].copy()

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
            model.low_vram_shift(is_diffusing=False)

        # Add more prompt keywords based on the semantics present
        if prompt_tokens['a_prompt'] is not None:
            a_prompt += ("," + prompt_tokens['a_prompt'])
        if prompt_tokens['n_prompt'] is not None:
            n_prompt += ("," + prompt_tokens['n_prompt'])

        cond = {"c_concat": [control], "c_crossattn": 
                [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                    "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if ControlNet.config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        model.control_scales = [strength * (0.825 ** float(12 - i)) \
                                for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if ControlNet.config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5)\
            .cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        seg_results = []
        for result in results:
            seg_results.append(preprocessor(resize_image(result, detect_resolution)))
    return [uniformer_segmap] + [detected_map] + results + seg_results

def get_ade20k(config):
    index_file = 'index_ade20k.pkl'
    dataset_path = os.path.join(os.getcwd(), config.image.DATASET_PATH)
    with open('{}/{}'.format(dataset_path, index_file), 'rb') as f:
        index_ade20k = pkl.load(f)
    
    file_name = index_ade20k['filename'][config.image.image_id]
    num_obj = index_ade20k['objectPresence'][:, config.image.image_id].sum()
    num_parts = index_ade20k['objectIsPart'][:, config.image.image_id].sum()
    count_obj = index_ade20k['objectPresence'][:, config.image.image_id].max()
    obj_id = np.where(index_ade20k['objectPresence'][:, config.image.image_id]\
                       == count_obj)[0][0]
    obj_name = index_ade20k['objectnames'][obj_id]
    full_file_name = '{}/{}'.format(index_ade20k['folder'][config.image.image_id],
                                     index_ade20k['filename'][config.image.image_id])

    print("The image at index {} is {}".format(config.image.image_id, file_name))
    print("It is located at {}".format(full_file_name))
    print("It happens in a {}".format(index_ade20k['scene'][config.image.image_id]))
    print("It has {} objects, of which {} are parts".format(num_obj, num_parts))
    print("The most common object is object {} ({}), which appears {} times".format(obj_name, obj_id, count_obj))

    info = utils_ade20k.loadAde20K('{}/{}'.format(config.image.DATASET_PATH, full_file_name))
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

def get_waymo_data(config):
    raise NotImplementedError

if __name__ == "__main__":
    config = omegaconf.OmegaConf.load('segment_unittest/config.yaml')
    model, ddim_sampler = initialize_model(config)
    img, seg_mask, object_keys = get_ade20k(config)

    prompt_tokens = {'a_prompt': '', 'n_prompt': ''}
    for key in object_keys:
        if key != '-':
            prompt_tokens['a_prompt'] += 'same {}, '.format(key)
            prompt_tokens['n_prompt'] += 'missing {}, '.format(key)

    if config.GRADIO:
        gradio_viz(title='contorl net test', 
                model=model, ddim_sampler=ddim_sampler,
                input_image=img.copy(), seg_mask = [seg_mask.copy()], 
                server_name='hg22723@swarmcluster1.ece.utexas.edu', server_port=8090,
                prompt_tokens=prompt_tokens)
    else:
        run_model(img.copy(), 
                model=model, ddim_sampler=ddim_sampler, seg_mask=[seg_mask.copy()])
# TODO: Modularization of this function is pending
# DESIGN_DICT =  [{'type': 'image', 
#                'title': 'Input Image', 
#                'key': 'input_image', 
#                'source': 'upload', 
#                'type': 'numpy'},
#                 {'type': 'textbox',
#                  'title': 'Prompt',
#                  'key': 'prompt'},