import mmcv
import mmengine
import matplotlib.pyplot as plt
import logging
import os.path as osp
import numpy as np

from avcv.dataset.dataset_wrapper import WaymoDatasetMM, AVResize
from mmengine import Config
from mmengine.runner import Runner
from omegaconf import OmegaConf
import os
from mmengine.logging import print_log
from argparse import ArgumentParser
import torch
class AVTrain:
    
    def __init__(self,
                 train_config,
                 args) -> None:
        self.train_config = train_config
        
        # cfg is the model config
        cfg = Config.fromfile(train_config.model_config_path)
        print(f'Config:\n{cfg.pretty_text}')
        cfg.work_dir = train_config.work_dir
        
        args = parse_args()

        # load config
        cfg.launcher = args.launcher

        # enable automatic-mixed-precision training
        if args.amp is True:
            optim_wrapper = cfg.optim_wrapper.type
            if optim_wrapper == 'AmpOptimWrapper':
                print_log(
                    'AMP training is already enabled in your config.',
                    logger='current',
                    level=logging.WARNING)
            else:
                assert optim_wrapper == 'OptimWrapper', (
                    '`--amp` is only supported when the optimizer wrapper type is '
                    f'`OptimWrapper` but got {optim_wrapper}.')
                cfg.optim_wrapper.type = 'AmpOptimWrapper'
                cfg.optim_wrapper.loss_scale = 'dynamic'

        # resume training
        cfg.resume = args.resume

        # Load the pretrained weights
        cfg.load_from = train_config.load_model_paths

        # Set up working dir to save files and logs.

        cfg.train_cfg.max_iters = train_config.max_iters
        cfg.train_cfg.val_interval = train_config.val_interval
        cfg.default_hooks.logger.interval = train_config.logger_interval
        cfg.default_hooks.checkpoint.interval = train_config.checkpoint_interval

        # Set seed to facilitate reproducing the result
        cfg['randomness'] = dict(seed=0)
        print(f'Config:\n{cfg.pretty_text}')
        self.engine = Runner.from_cfg(cfg)
        
       
    def train(self):
        self.engine.train()

def parse_args():
    parser = ArgumentParser(description='Train a segmentor')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

    
if __name__ == '__main__':

    train_config = OmegaConf.load('avcv/configs/train_configs/mask2former_r50_8xb2-90k_waymo-512x512.yaml')
    args = parse_args()
    trainer = AVTrain(train_config, args)
    trainer.train()