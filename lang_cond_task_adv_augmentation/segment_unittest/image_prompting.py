import argparse
import torch
import sys
import os

sys.path.append(os.path.join(os.getcwd(),'LLaVA/'))
from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO

from segment_unittest.utils import convert_pallette_segment
import omegaconf
from waymo_open_data_parser.data_loader import WaymoDataset

def load_image(dataset: WaymoDataset, index):
    image, _, _, object_mask = dataset[index]
    prompt_tokens = dataset.get_text_description(object_mask)
    return image, prompt_tokens

def get_caption(model: torch.nn.Module,
                model_name: str,
                image_processor: torch.nn.Module,
                tokenizer: torch.nn.Module,
                prompt: str,
                dataset: WaymoDataset ,
                index: int,
                query: str,
                conv_mode: str):

    image, prompt_objects = load_image(dataset, index)
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
    
    return outputs


def caption_dataset(config):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(config.LLAVACAPTION.MODELPATH)
    if config.LLAVACAPTION.MODELBASE =='None':
        config.LLAVACAPTION.MODELBASE = None
    tokenizer, model, image_processor, context_len \
        = load_pretrained_model(config.LLAVACAPTION.MODELPATH, 
                                config.LLAVACAPTION.MODELBASE,
                                  model_name)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if config.LLAVACAPTION.conv_mode is not None \
        and conv_mode != config.LLAVACAPTION.conv_mode:
        print('[WARNING] the auto inferred conversation mode is\
               {}, while `--conv-mode` is {}, using {}'.format(conv_mode, 
                                                    config.LLAVACAPTION.conv_mode, 
                                                    config.LLAVACAPTION.conv_mode))
    else:
        config.LLAVACAPTION.conv_mode = conv_mode

    dataset = WaymoDataset(config.image)
    captions = []
    for j in range(config.LLAVACAPTION.NUM_IMAGES):
        caption = get_caption(model, 
                              model_name, 
                              image_processor, 
                              tokenizer, 
                              config.LLAVACAPTION.prompt, 
                              dataset, 
                              j, 
                              config.LLAVACAPTION.prompt, 
                              config.LLAVACAPTION.conv_mode)
        print(caption)
        captions.append(caption)
    
    return captions

if __name__ == "__main__":
    config = omegaconf.OmegaConf.load("segment_unittest/config.yaml")
    captions = caption_dataset(config)

    # Store the captions somewhere
    with open("captions.txt", "w") as f:
        for caption in captions:
            f.write(caption)
            f.write("\n")