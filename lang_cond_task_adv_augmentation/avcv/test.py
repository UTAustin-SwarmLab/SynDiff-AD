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
import pandas as pd
from mmengine.runner  import autocast
from tqdm import tqdm
from lang_data_synthesis.utils import write_to_csv_from_dict, to_numpy, to_cpu
from prettytable import PrettyTable
from collections import OrderedDict
# This code would load the models and characterise the tests based on
# the type of test condition.


# Load the test conditions from waymo_open_data/waymo_conditions_val.csv
class ConditionAVTester:
    
    def __init__(self,
                test_config,
                args) -> None:
    
        self.test_config = test_config
        
        if 'synth' in test_config.model_name \
            or 'mixed' in test_config.model_name \
                or 'ft' in test_config.model_name:
            model_config_path = test_config.model_config_path + test_config.source_model + '.py'
        else:
            model_config_path = test_config.model_config_path + test_config.model_name + '.py'
            
        cfg = Config.fromfile(model_config_path)
        print(f'Config:\n{cfg.pretty_text}')
        cfg.work_dir = test_config.work_dir + test_config.model_name + args.dir_tag
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

        # Loads Checkpoint

        loads = cfg.work_dir 
        latest_path = 0
        latest_iter = 0
        for i in os.listdir(loads):
            if i.endswith('.pth') and i.startswith('best'):
                #find iter number
                iter = int(i.split('.')[0].split('_')[-1])
                if iter>latest_iter:
                    latest_iter = iter
                    latest_path = i
        
        cfg.resume = True
        cfg.load_from = osp.join(loads, latest_path)

        # TODO: Load the dataset stuff

        if 'bdd' in test_config.model_name:
            FILENAME = self.test_config.data_path + "metadata_val_seg.csv"
        elif 'waymo' in test_config.model_name:
            FILENAME = self.test_config.data_path + "waymo_env_conditions_val.csv"

        if os.path.exists(FILENAME):
            self.metadata_conditions = pd.read_csv(FILENAME)
            print(self.metadata_conditions.columns)
            self.dataset_length = len(self.metadata_conditions)
            print(self.metadata_conditions\
                .groupby(['condition']).size() / self.dataset_length * 100)
            
        self.engine = Runner.from_cfg(cfg)
        self.engine.load_or_resume()
        self.cfg = cfg
        
        if 'bdd' in test_config.model_name:
             data_dict = {
                "file_name":"file_name",
                "condition":"condition"
            }
        elif 'waymo' in test_config.model_name:
            data_dict = {
                "context_name":"context_name",
                "context_frame":"context_frame",
                "camera_id":"camera_id",
                "condition":"condition"
            }
        for metric in self.engine.test_evaluator.metrics:
            data_dict[metric.metrics[0]+'_intersect'] = metric.metrics[0]+'_intersect'
            data_dict[metric.metrics[0]+'_union'] = metric.metrics[0]+'_union'
            data_dict[metric.metrics[0]+'_pred_label'] = metric.metrics[0]+'_pred_label'
            data_dict[metric.metrics[0]+'_label'] = metric.metrics[0]+'_label'

            
        self.save_filename = test_config.test_data_path  + test_config.model_name + ".csv"
        write_to_csv_from_dict(
                    dict_data=data_dict , 
                    csv_file_path= self.save_filename,
                    file_name=""
        )
    
    def test(self):
        # TODO: TEST THE dataset with respect to different conditions
        # obtain the metadata from the runner's validation dataloader, 
        # invoke the condition of the respective frames stores within the classification
        # dataset.
        # Obtain the test results, per object mIOU, per object accuracy, per object 
        # precision, per object recall
        # average recall, etc and store the results in .csv file.
        # Define new metric vIOU per test condition
        dataloader = self.engine.test_dataloader
        evaluator = self.engine.test_evaluator
        self.engine.model.eval()
        
        with torch.no_grad():
            for idx, data_batch in enumerate(tqdm(dataloader, desc='val', total=len(dataloader))):
                # Lets resolve the tester here
                
                with autocast(enabled=self.engine.val_loop.fp16):
                    outputs = self.engine.model.val_step(data_batch)
                    
                evaluator.process(data_samples=outputs, 
                                data_batch=data_batch)
                
                
                # Obtain the condition from the dataset
                for j,sample in enumerate(data_batch['data_samples']):

                    
                    # weather = condition.split(',')[-1]
                    # condition = sample.condition
                    # time = condition.split(',')[-1]
                    # condition = weather + ',' + time
                    
                    if 'bdd' in self.test_config.model_name:
                        print(sample.keys())
                        file_name = sample.file_name
                        
                        condition = self.metadata_conditions.loc[(
                            self.metadata_conditions['file_name'] == file_name)].condition.values[0]
                        data_dict = {
                            'file_name':sample.file_name,
                            'condition':condition 
                        }
                    elif 'waymo' in self.test_config.model_name:
                        context_name = sample.context_name
                        context_frame = sample.context_frame
                        camera_id = sample.camera_id
                        
                        condition = self.metadata_conditions.loc[(self.metadata_conditions['context_name'] == context_name) & \
                            (self.metadata_conditions['context_frame'] == context_frame) & \
                            (self.metadata_conditions['camera_id'] == camera_id)].condition.values[0]
                        
                        data_dict = {
                            'context_name':context_name,
                            'context_frame':context_frame,
                            'camera_id':camera_id,
                            'condition':condition 
                        }
                    
                    for metric in evaluator.metrics:
                        results = [[to_numpy(a) for a in metric.results[j]]]
                        results  = tuple(zip(*results))
                        total_area_intersect = sum(results[0])
                        total_area_union = sum(results[1])
                        total_area_pred_label = sum(results[2])
                        total_area_label = sum(results[3])

                        data_dict[metric.metrics[0]+'_intersect'] = total_area_intersect
                        data_dict[metric.metrics[0]+'_union'] = total_area_union
                        data_dict[metric.metrics[0]+'_pred_label'] = total_area_pred_label
                        data_dict[metric.metrics[0]+'_label'] = total_area_label
                        
                        # ret_metrics = metric.total_area_to_metrics(
                        #     total_area_intersect, total_area_union, total_area_pred_label,
                        #     total_area_label, metric.metrics,
                        #     metric.nan_to_num, metric.beta)
                        
                        # class_names = metric.dataset_meta['classes']

                        # # summary table
                        # ret_metrics_summary = OrderedDict({
                        #     ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                        #     for ret_metric, ret_metric_value in ret_metrics.items()
                        # })
                        
                        # metrics = dict()
                        # for key, val in ret_metrics_summary.items():
                        #     if key == 'aAcc':
                        #         metrics[key] = val
                        #     else:
                        #         metrics['m' + key] = val

                        # # each class table
                        # ret_metrics.pop('aAcc', None)
                        # ret_metrics_class = OrderedDict({
                        #     ret_metric: np.round(ret_metric_value * 100, 2)
                        #     for ret_metric, ret_metric_value in ret_metrics.items()
                        # })
                        # ret_metrics_class.update({'Class': class_names})
                        # ret_metrics_class.move_to_end('Class', last=False)
                        
                        # data_dict[metric.metrics[0]+'_metric'] = metrics
                        # data_dict[metric.metrics[0]+'_perclass'] = dict(ret_metrics_class)
                    
                    write_to_csv_from_dict(
                        dict_data=data_dict , 
                        csv_file_path= self.save_filename,
                        file_name=""
                    )
                for metric in evaluator.metrics:
                    metric.results.clear()
                torch.cuda.empty_cache()

        return None
    
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
        default='segformer_mit-b3_8xb2-160k_bdd-512x512',
        help='train config in configs/test_configs/'
    )
    parser.add_argument(
        '--dir_tag',
        default='v1',
        help='tag for new directory to save model'
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
    config_path = 'avcv/configs/test_configs/' + args.config_name + '.yaml'
    test_config = OmegaConf.load(
        config_path
    )
    tester = ConditionAVTester(test_config, args)
    tester.test()