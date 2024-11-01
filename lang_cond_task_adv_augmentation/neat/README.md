# NEAT: Neural Attention Fields for End-to-End Autonomous Driving

## [Paper](http://www.cvlibs.net/publications/Chitta2021ICCV.pdf) | [Supplementary](http://www.cvlibs.net/publications/Chitta2021ICCV_supplementary.pdf) | [Video](https://www.youtube.com/watch?v=gtO-ghjKkRs) | [Talk](https://www.youtube.com/watch?v=hYm6LPTyHHA) | [Poster](https://www.cvlibs.net/publications/Chitta2021ICCV_poster.pdf) | [Slides](https://www.cvlibs.net/publications/Chitta2021ICCV_slides.pdf)

<img src="neat/assets/neat_clip.GIF" height="270" hspace=30>

This repository is for the ICCV 2021 paper [NEAT: Neural Attention Fields for End-to-End Autonomous Driving](http://www.cvlibs.net/publications/Chitta2021ICCV.pdf).

```bibtex
@inproceedings{Chitta2021ICCV,
  author = {Chitta, Kashyap and Prakash, Aditya and Geiger, Andreas},
  title = {NEAT: Neural Attention Fields for End-to-End Autonomous Driving},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year = {2021}
}
```

## Setup
Please follow the installation instructions from our [TransFuser repository](https://github.com/autonomousvision/transfuser/tree/cvpr2021) to set up the CARLA simulator. The conda environment required for NEAT can be installed via:
```Shell
conda env create -f environment.yml
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

For running the AIM-VA baseline, you will additionally need to install [MMCV](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation).
```Shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmsegmentation
```

## Data Generation
The training data is generated using ```leaderboard/team_code/auto_pilot.py```. Data generation requires routes and scenarios. Each route is defined by a sequence of waypoints (and optionally a weather condition) that the agent needs to follow. Each scenario is defined by a trigger transform (location and orientation) and other actors present in that scenario (optional). We provide several routes and scenarios under ```leaderboard/data/```. The [TransFuser repository](https://github.com/autonomousvision/transfuser) and [leaderboard repository](https://github.com/carla-simulator/leaderboard/tree/master/data) provide additional routes and scenario files.

### Running a CARLA Server

#### With Display
```Shell
./CarlaUE4.sh --world-port=2000 -opengl
```

#### Without Display

Without Docker:
```Shell
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./CarlaUE4.sh --world-port=2000 -opengl
```

With Docker:

Instructions for setting up docker are available [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Pull the docker image of CARLA 0.9.10.1 ```docker pull carlasim/carla:0.9.10.1```.

Docker 18:
```Shell
docker run -it --rm -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.9.10.1 ./CarlaUE4.sh --world-port=2000 -opengl
```

Docker 19:
```Shell
docker run -it --rm --net=host --gpus '"device=0"' carlasim/carla:0.9.10.1 ./CarlaUE4.sh --world-port=2000 -opengl
```

If the docker container doesn't start properly then add another environment variable ```-e SDL_AUDIODRIVER=dsp```.

### Running the Autopilot

Once the CARLA server is running, rollout the autopilot to start data generation.
```Shell
./leaderboard/scripts/run_evaluation.sh
```
The expert agent used for data generation is defined in ```leaderboard/team_code/auto_pilot.py```. Different variables which need to be set are specified in ```leaderboard/scripts/run_evaluation.sh```. The expert agent is originally based on the autopilot from [this codebase](https://github.com/bradyz/2020_CARLA_challenge).

## Training
The training code and pretrained models are provided below.
```Shell
mkdir model_ckpt
wget https://s3.eu-central-1.amazonaws.com/avg-projects/neat/models.zip -P model_ckpt
unzip model_ckpt/models.zip -d model_ckpt/
rm model_ckpt/models.zip
```

There are 5 pretrained models provided in ```model_ckpt/```:
- [AIM-MT (2D)](leaderboard/team_code/aim_mt_2d_agent.py): ```aim_mt_sem``` and ```aim_mt_sem_depth```
- [AIM-MT (BEV)](leaderboard/team_code/aim_mt_bev_agent.py): ```aim_mt_bev```
- [AIM-VA](leaderboard/team_code/aim_va_agent.py): ```aim_va```
- [NEAT](leaderboard/team_code/neat_agent.py): ```neat```

Additional baselines are available in the [TransFuser repository](https://github.com/autonomousvision/transfuser).

## Evaluation
Spin up a CARLA server (described above) and run the required agent. The required variables need to be set in ```leaderboard/scripts/run_evaluation.sh```.
```Shell
CUDA_VISIBLE_DEVICES=0 ./leaderboard/scripts/run_evaluation.sh
```

## Acknowledgements
This implementation primarily extends the [cvpr2021 branch of the existing TransFuser repository](https://github.com/autonomousvision/transfuser/tree/cvpr2021).

If you found our work interesting, check out the code for some more recent work on CARLA from our group:
- [Renz et al., PlanT: Explainable Planning Transformers via Object-Level Representations (CoRL 2022)](https://github.com/autonomousvision/plant)
- [Chitta et al., TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving (PAMI 2022)](https://github.com/autonomousvision/transfuser)
- [Hanselmann et al., KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients (ECCV 2022)](https://github.com/autonomousvision/king)
