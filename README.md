# SynDiff-AD

1. Please set pythonpath during execution to the lang_cond_task_adv_augmentation folder in the repo before executing scripts see 8.

3. Development
   ```
   conda env create -n control-v11.8 -f environment.yml
   ```
   Use the conda env to develop further. Ensure torch version is 2.1.0 and cuda 11.8 by running 
   conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

6.  To install and setup mmsegmentation
   ```
   pip install -U openmim
   mim install mmengine==0.10.3
   mim install "mmcv>=2.1.0"
   mim install "mmdet==3.3.0"
   mim install "mmsegmentation>=1.0.0"
   ```
   Ignore the dependecy conflict of numpy 1.21.5 and pip install numpy==1.23.1- works seamlessly

7. Use the following commands to download the model weights for different segmentation models [higly recommended to finetune models]

   ```
   mim download mmsegmentation --config mask2former_swin-t_8xb2-160k_ade20k-512x512 --dest avcv/model_weights
   ```
Replace the  --config with any model config you want installed. Refer to https://github.com/open-mmlab/mmsegmentation/tree/main/configs for more configs.

8. Training models with mmseg. We provide a sample command for training a model config with mmseg.

First locate a model config from avcv/configs/models
Confirm .yaml config exists for your selected model in avcv/config/train_configs

   ```
   PYTHONPATH=/home/hg22723/projects/lang-cond-task-adv-augmentation/lang_cond_task_adv_augmentation CUDA_VISIBLE_DEVICES=0,1 python3 train.py --config mask2former_swin-t_8xb2-160k_ade20k-512x512 
   ```

9. File structure for using lang_data_synthesis

Scripts not mentioned here are Work in Progress

a - image_classification.py - classifies the images as per test conditions specified in lang_data_synthesis/config.yaml

b - image_prompting.py - captioning images using LLAVA-1.5 upgrade pending

c - imgseg_synthesis.py - single image conditional synthesis using controlnet

d - utils.py - utility functions

e - synthesis.py - synthesizes a dataset using controlnet and the target conditions

f - test_synthesis.py - tests the synthesised dataset to produce FID scores with respect to different backbones

10. Since the model weights have not been pushed, the results file are available
  -  avcv/tests comprises all the results for finetuning
  -  Run the results.ipynb script in avcv/experiments to view the compiled results
  - Since the synthetic dataset needs to be produced, checkout the .json files under lang_data_synthesis/synthesis_results for FID scores



 
