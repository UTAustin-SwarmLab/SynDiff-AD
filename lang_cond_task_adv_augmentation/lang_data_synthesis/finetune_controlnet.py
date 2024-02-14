import ControlNet.config:
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'ControlNet/'))
from ControlNet.cldm.hack import disable_verbosity, enable_sliced_attention


disable_verbosity()

if ControlNet.config.save_memory:
    enable_sliced_attention()


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lang_data_synthesis.dataset import AVControlNetDataset
from ControlNet.cldm.logger import ImageLogger
from ControlNet.cldm.model import create_model, load_state_dict

from omegaconf import OmegaConf
from argparse import ArgumentParser
import torch
import os
import tensorflow as tf
class ControlNetFineTune:
    
    def __init__(self, config, args) -> None:
        
        self.config = config
        self.args = args
        if not os.path.exists(config.path_init_ckpt):
            self.model = self.setup_checkpoint(config)
        else:
            self.model = create_model(config.config_path_sd15).cpu()
            self.model.load_state_dict(load_state_dict(config.path_init_ckpt), 
                                                    strict = False)

            
        self.model.learning_rate = config.learning_rate
        self.model.sd_locked = config.sd_locked
        self.model.only_mid_control = config.only_mid_control

    def setup_checkpoint(self, 
                         config):
        
        model = create_model(config.config_path_sd15).cpu()
        model.load_state_dict(load_state_dict(config.path_sd15), 
                                                strict = False)
        model.load_state_dict(
            load_state_dict(config.path_sd15_with_control),
            strict = False)

        torch.save(model.state_dict(), config.path_init_ckpt)
        print('Transferred model saved at ' + config.path_init_ckpt)
        return model
    
    def finetune_model(self):
        # Misc

        if self.args.experiment == 'seg':
            configs = {
                'waymo' : OmegaConf.load('lang_data_synthesis/waymo_config.yaml'),
                'bdd' : OmegaConf.load('lang_data_synthesis/bdd_config.yaml')
            }
        elif self.args.experiment == 'carla':
            configs = {
                'carla': OmegaConf.load('lang_data_synthesis/carla_config.yaml')
            }
    
        # Switch
        # Add wandb callback logger
        # wandb_logger = pl.loggers.WandbLogger(project='controlnet', save_dir='lang_data_synthesis/finetune_logs/')
        # wandb_logger.watch(self.model)
        
        
        dataset = AVControlNetDataset(configs, rare_class_module=self.args.rct)
        dataloader = DataLoader(dataset, num_workers=self.args.num_workers,
                                batch_size=self.config.batch_size,
                                shuffle=True)

        logger = ImageLogger(batch_frequency=self.config.logger_freq)
        trainer = pl.Trainer(gpus=self.args.num_gpus, precision=32, callbacks=[logger])#, logger= wandb_logger)


        # Train!
        trainer.fit(self.model, dataloader)

def parse_args():
    parser = ArgumentParser(description='Image Classification with CLIP')

    parser.add_argument(
        '--rct',
        action='store_true',
        default=False,
        help='Whether to use rare class training')
    
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=1,
        help='Number of gpus to use for training')

    parser.add_argument(
        '--experiment',
        choices=['seg','carla', 'cliport'],
        default='none',
        help='Which experiment config to generate data for')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,
        help='Number of workers per gpu for loading data')
    return  parser.parse_args()


if  __name__ == "__main__":
    args = parse_args()
    tf.config.set_visible_devices([], 'GPU')
    config = OmegaConf.load('lang_data_synthesis/finetune_config.yaml')

    finetune = ControlNetFineTune(config, args)
    finetune.finetune_model()
