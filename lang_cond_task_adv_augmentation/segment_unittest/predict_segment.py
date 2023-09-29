# Need segmentation maks from SAM for almost near ground truth classification
from segment_unittest.generate_image_with_segment import process
from typing import *
import numpy as np

TEST_PROMPTS = ["Night, good lighting",
                "Night, bad lighting",
                "Foggy, good lighting",
                "Foggy, bad lighting",
                "Rainy, good lighting",
                "Rainy, bad lighting"]

NUM_GENERATED_IMAGES = 10

def make_dataset(
        input_image: np.ndarray,
        seg_mask: np.ndarray,
        model: Any,
        ddim_sampler: Any,
        config: dict,
        prompt_tokens:dict
        ):
    '''
    Returns a list of synthetic 
    images with the dataset conditioned on the segmentation images and itself
    based on the TEST PROMPT conditions
    '''
    outputs = {prompt:[] for prompt in TEST_PROMPTS}
    for prompt in TEST_PROMPTS:
        for j in range(NUM_GENERATED_IMAGES//config.SD_GEN_PARAMS.NUMSAMPLES):
            print(f"Generating image {j} for prompt {prompt}")
            out = process(input_image, prompt, 
                            config.SD_GEN_PARAMS.A_PROMPT,
                            config.SD_GEN_PARAMS.N_PROMPT,
                            config.SD_GEN_PARAMS.NUMSAMPLES, 
                            config.SD_GEN_PARAMS.IMAGE_RESOLUTION,
                            config.SD_GEN_PARAMS.DETECT_RESOLUTION,
                            config.SD_GEN_PARAMS.DDIMSTEPS, 
                            config.SD_GEN_PARAMS.GUESS, 
                            config.SD_GEN_PARAMS.STRENGTH, 
                            config.SD_GEN_PARAMS.SCALE,
                            config.SD_GEN_PARAMS.SEED, 
                            config.SD_GEN_PARAMS.ETA,
                            det=config.SD_GEN_PARAMS.ANNOTATOR, 
                            model=model, 
                            ddim_sampler=ddim_sampler,
                            seg_mask=seg_mask, 
                            prompt_tokens=prompt_tokens)
            num_out = len(out)
            outputs[prompt] += out[2:3+(num_out-2)//2]
        outputs[prompt] = np.concatenate(outputs[prompt], axis=0)
    return outputs

