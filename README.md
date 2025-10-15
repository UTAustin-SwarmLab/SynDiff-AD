
# SynDiff-AD: Synthetic Data Generation for Autonomous Driving with Diffusion Models

**[ArXiv] | [Paper]**


`SynDiff-AD` is a framework leveraging conditional diffusion models to generate high-fidelity, diverse synthetic data for autonomous driving applications. By synthesizing images from semantic layouts, we enhance the training of downstream perception models, such as semantic segmentation, improving their robustness and performance. This repository provides the official implementation for the SynDiff-AD data generation pipeline and model training.

!(placeholder_for_your_image.png)

---

## ⚙️ Setup

This project was developed using `Python 3.8+`, `PyTorch 2.1.0`, and `CUDA 11.8`.

### 1. Clone Repository
First, clone the project repository to your local machine:
```bash
git clone [https://github.com/your-username/SynDiff-AD.git](https://github.com/your-username/SynDiff-AD.git)
cd SynDiff-AD
```



### 2. Development
Install project requirements
```
conda env create -n control-v11.8 -f environment.yml
```

Use the conda env to develop further. Ensure torch version is 2.1.0 and cuda 11.8.

```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install and setup mmsegmentation for segmentation experiments

```
   pip install -U openmim
   mim install mmengine==0.10.3
   mim install "mmcv>=2.1.0"
   mim install "mmdet==3.3.0"
   mim install "mmsegmentation>=1.0.0"

```
Note: Ignore the dependency conflict of numpy 1.21.5 and pip install numpy==1.23.1- works seamlessly


## Synthetic Data Generation

The core of this project is the language-driven data synthesis pipeline located in the lang_data_synthesis directory.

Key Scripts:

image_classification.py: Classifies images according to the test conditions specified in lang_data_synthesis/config.yaml.

image_prompting.py: Captions images using LLaVA-1.5 (upgrade pending).

imgseg_synthesis.py: Performs conditional synthesis on a single image using ControlNet.

utils.py: Contains various utility functions for the pipeline.

synthesis.py: The main script to synthesize an entire dataset using ControlNet and the specified target conditions.

test_synthesis.py: Tests a synthesized dataset to produce FID scores with respect to different backbones.

### Example: Use the synthesis script with your CARLA data
python lang_data_synthesis/synthesis.py --input_dir data/carla_raw --output_dir data/carla_synthetic --config your_synthesis_config.yaml


## Training Segmentation Models

### 1. Download Pre-trained Models
We use pre-trained models from mmsegmentation as a starting point for fine-tuning. We highly recommend downloading the model weights before proceeding.

Use the following command to download weights for a specific model, for example, mask2former_swin-t. The weights will be saved in avcv/model_weights [higly recommended to finetune models]

```
mim download mmsegmentation --config mask2former_swin-t_8xb2-160k_ade20k-512x512 --dest avcv/model_weights
```
Replace the  --config with any model config you want installed. Refer to https://github.com/open-mmlab/mmsegmentation/tree/main/configs for more configs.

### 2. Training Models

We provide scripts for fine-tuning segmentation models using mmsegmentation.
Locate a model config file from avcv/configs/models.
Ensure a corresponding training config .yaml file exists in avcv/config/train_configs.
Run the training script.

Here is a sample command to train a mask2former_swin-t model on two GPUs (0 and 1):
Please set pythonpath during execution to the lang_cond_task_adv_augmentation folder in the repo before executing scripts see 8.

```
   PYTHONPATH=/home/hg22723/projects/lang-cond-task-adv-augmentation/lang_cond_task_adv_augmentation CUDA_VISIBLE_DEVICES=0,1 python3 train.py --config mask2former_swin-t_8xb2-160k_ade20k-512x512 
```

### 3. Results

Since the model weights have not been pushed, the results file are available
  -  avcv/tests comprises all the results for finetuning
  -  Run the results.ipynb script in avcv/experiments to view the compiled results



## Training Autonomous Driving models for CARLA

This section outlines the process for generating data with the CARLA simulator and using it with our pipeline.


### 1. CARLA Setup
Ensure you have CARLA installed and configured. Please follow the official CARLA documentation for installation.

### 2. Data Collection
Use our provided scripts to connect to a running CARLA instance and collect semantic segmentation maps and corresponding RGB images.

Bash

# Example command to collect 1000 frames from CARLA
python carla_utils/collect_data.py --host localhost --port 2000 --frames 1000 --output-dir data/carla_raw

### 3. Training

### 4. Testing



### Results

Since the synthetic dataset needs to be produced, checkout the .json files under lang_data_synthesis/synthesis_results for FID scores

## Citation

```
@article{goel2024syndiff,
  title={Syndiff-ad: Improving semantic segmentation and end-to-end autonomous driving with synthetic data from latent diffusion models},
  author={Goel, Harsh and Narasimhan, Sai Shankar and Akcin, Oguzhan and Chinchali, Sandeep},
  journal={arXiv preprint arXiv:2411.16776},
  year={2024}
}
```
```
@inproceedings{
goel2024improving,
title={Improving End-To-End Autonomous Driving with Synthetic Data from Latent Diffusion Models},
author={Harsh Goel and Sai Shankar Narasimhan and Sandeep P. Chinchali},
booktitle={First Vision and Language for Autonomous Driving and Robotics Workshop},
year={2024},
url={https://openreview.net/forum?id=yaXYQinjOA}
}
```

 
