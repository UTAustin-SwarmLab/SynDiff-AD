import mmcv
import mmengine
import matplotlib.pyplot as plt
import logging
import os.path as osp
import numpy as np

from avcv.dataset.dataset_wrapper import WaymoDatasetMM, AVResize
from avcv.dataset import *
from avcv.dataset.synth_dataset_wrapper import SynthWaymoDatasetMM
from avcv.dataset.mixed_dataset_wrapper import MixedWaymoDatasetMM
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
        model_config_path = train_config.model_config_path + train_config.model_name + '.py'
        cfg = Config.fromfile(model_config_path)
        print(f'Config:\n{cfg.pretty_text}')
        

        cfg.work_dir = train_config.work_dir + train_config.model_name + args.dir_tag + args.add_tag
            
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
        if args.resume:
            loads = cfg.work_dir 
            latest_path = 0
            latest_iter = 0
            for i in os.listdir(loads):
                if i.endswith('.pth'):
                    #find iter number
                    iter = int(i.split('.')[0].split('_')[-1])
                    if iter>latest_iter:
                        latest_iter = iter
                        latest_path = i
            cfg.load_from = osp.join(loads, latest_path)
            #cfg.load_from = osp.join(loads, 'last_checkpoint')
        elif 'synth' in train_config.model_name or 'mixed' in train_config.model_name or 'ft-' in  train_config.model_name or 'aug-' in train_config.model_name :
            loads = train_config.work_dir + train_config.source_model + args.dir_tag
            latest_path = 0
            latest_iter = 0
            for i in os.listdir(loads):
                if i.endswith('.pth'):
                    #find iter number
                    iter = int(i.split('.')[0].split('_')[-1])
                    if iter>latest_iter:
                        latest_iter = iter
                        latest_path = i
            cfg.load_from = osp.join(loads, latest_path)
        else:
            cfg.load_from = train_config.load_model_paths 
        


        # Set up working dir to save files and logs.
        cfg.train_cfg.max_iters = train_config.max_iters
        cfg.train_cfg.val_interval = train_config.val_interval
        
          
        schedulers = []
        for param_scheduler in cfg.param_scheduler:
            param_scheduler.end = train_config.max_iters
            schedulers.append(param_scheduler)
        cfg.param_scheduler = schedulers
            
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
    parser.add_argument(
        '--config_name',
        default='mask2former_r50_8xb2-90k_mixedwaymo-512x512',
        help='train config in configs/train_configs/'
    )
    parser.add_argument(
        '--dir_tag',
        default='',
        help='tag for new directory to save model'
    )
    parser.add_argument(
        '--add_tag',
        default='',
        help='tag for new directory to save model for finetuned models'
    )

    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

    
if __name__ == '__main__':
    args = parse_args()
    config_path = 'avcv/configs/train_configs/' + args.config_name + '.yaml'
    train_config = OmegaConf.load(
        config_path
    )
    trainer = AVTrain(train_config, args)
    trainer.train()