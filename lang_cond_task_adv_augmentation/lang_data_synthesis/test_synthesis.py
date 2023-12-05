# Computes per condition and total FID scores and stores it in a csv file

from typing import Any
from torchmetrics import Metric
from torch import Tensor
from copy import deepcopy
import torch


from torchvision.models import resnet50
import os
import json
import pandas as pd
import omegaconf
from mmseg.registry import DATASETS, TRANSFORMS
from avcv.dataset.dataset_wrapper import *
from avcv.dataset.synth_dataset_wrapper import SynthWaymoDatasetMM
from mmengine.registry import init_default_scope
from mmseg.models.data_preprocessor import SegDataPreProcessor
from tqdm import tqdm
init_default_scope('mmseg')

def _compute_fid(mu1: Tensor,
                sigma1: Tensor, 
                mu2: Tensor, sigma2
                : Tensor) -> Tensor:
    """Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.

    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return a + b - 2 * c

class FID_Metric(Metric):
    """
    FID metric implementation for PyTorch Lightning.
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    real_features_loaded: bool = False

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    def __init__(
        self,
        num_features: int,
        reset_real_features: bool = True,
    ):
        super().__init__()

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        mx_num_feets = (num_features, num_features)
        
        self.add_state(
            "real_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_num_feets).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_num_feets).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

    def update(self, features: Tensor, real: bool) -> None:
        """
        Updates the metric state.
        Args:
            features: Tensor containing features of generated images
            real: Boolean indicating if features are from real or generated images
        """

        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += features.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += features.shape[0]

    def compute(self) -> Tensor:
        """Calculate FID score based on stored features."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError(
                "More than one sample is required for both the real and fake distributed to compute FID"
            )
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(
            0
        )
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(
            0
        )

        cov_real_num = (
            self.real_features_cov_sum
            - self.real_features_num_samples * mean_real.t().mm(mean_real)
        )
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = (
            self.fake_features_cov_sum
            - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(
            mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake
        ).to(self.orig_dtype)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()

    def num_samples(self) -> int:
        """Return number of samples used to compute FID both fake and real"""
        return self.real_features_num_samples, self.fake_features_num_samples

def collate_fn(batch):
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        if key == 'inputs':
            collated_batch[key] = torch.stack([sample[key] for sample in batch])
        else:
            collated_batch[key] = [sample[key] for sample in batch]
    return collated_batch

class EvaluateSynthesisFID:

    def __init__(self,
                 config) -> None:
        # Load the data from the config : we load both the real and fake data
        # with the corresponding environment conditions

        # Create the datasets for both the fake data and real data
        self.config = config
        
        self.data_preprocessor = SegDataPreProcessor(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=False,
            pad_val=0,
            seg_pad_val=255,
            size=[512,512],
            test_cfg=dict(size_divisor=32)
        )
        
        self.real_data = WaymoDatasetMM(
            data_config=dict(
                TRAIN_DIR = self.config.IMAGE.WAYMO.TRAIN_DIR , 
                EVAL_DIR =  self.config.IMAGE.WAYMO.EVAL_DIR,
                TEST_SET_SOURCE = self.config.IMAGE.WAYMO.TEST_SET_SOURCE,
                SAVE_FRAMES = [0,1,2]),
            pipeline=[
                dict(type='AVResize', scale=[512,512], keep_ratio=False),
                dict(type='PackSegInputs', meta_keys=['context_name',
                                                    'context_frame',
                                                    'camera_id',
                                                    'ori_shape',
                                                    'img_shape',
                                                    'scale_factor',
                                                    'reduce_zero_label'])],
                validation=False,
                segmentation=True,
                image_meta_data=True,
                serialize_data=True
        )
        
        self.real_data_loader = torch.utils.data.DataLoader(
            self.real_data,
            batch_size=16,
            num_workers=16,
            persistent_workers=True,
            shuffle=False,
            collate_fn=collate_fn
        )
        # Create per_class FID metrics and full dataset FID metric
        self.fake_data = SynthWaymoDatasetMM(
            data_config=dict(
                DATASET_DIR = self.config.SYN_DATASET_GEN.dataset_path,
            ),
            pipeline=[
                dict(type='AVResize', scale=[512,512], keep_ratio=False),
                dict(type='PackSegInputs', meta_keys=['file_name',
                                                    'ori_shape',
                                                    'img_shape',
                                                    'scale_factor',
                                                    'reduce_zero_label'])],
                validation=False,
                segmentation=True,
                image_meta_data=True,
                serialize_data=True
        )
        self.fake_data_loader = torch.utils.data.DataLoader(
            self.fake_data,
            batch_size=16,
            num_workers=16,
            persistent_workers=True,
            shuffle=False,
            collate_fn=collate_fn
        )
        # Initialize feature extraction backbone RESNET 50 and the data preprocessor
        
        self.feature_extractor = resnet50(weights='DEFAULT')
        
        # Initialize per class FID metrics
        self.fid_dict = {}
        TRAIN_FILENAME = self.config.SYN_DATASET_GEN.conditions_path + "waymo_env_conditions_train.csv"
        if os.path.exists(TRAIN_FILENAME):
            self.real_metadata_conditions = pd.read_csv(TRAIN_FILENAME)
            self.conditions = self.real_metadata_conditions['condition'].unique()
        
        FAKE_FILENAME = self.config.SYN_DATASET_GEN.dataset_path + "/metadata_seg.csv"
        if os.path.exists(FAKE_FILENAME):
            self.fake_metadata_conditions = pd.read_csv(FAKE_FILENAME)
            
        # Load the environment conditions of the real data
        for condition in self.conditions:
            self.fid_dict[condition] = FID_Metric(num_features=1000)
        self.final_fid = FID_Metric(num_features=1000)
    
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
    
        with torch.no_grad():
            for j,data in tqdm(enumerate(self.real_data_loader), total=len(self.real_data_loader)):
                # if j>=20:
                #     break
                data = self.data_preprocessor(data)
                #data = self.real_data[j]
                images = data['inputs']
                data_samples = data['data_samples']
                # Extract features from the real data
                
                real_features = self.feature_extractor(images)
                
                for i in range(0, len(data_samples)):
                    meta_data = data_samples[i].metainfo
                    condition = self.real_metadata_conditions.loc[
                        (self.real_metadata_conditions['context_name'] == meta_data['context_name']) & 
                        (self.real_metadata_conditions['context_frame'] == meta_data['context_frame']) & 
                        (self.real_metadata_conditions['camera_id'] == meta_data['camera_id']) 
                    ]['condition'].values[0]
                    # Update the FID metric with the features
                    self.fid_dict[condition].update(real_features[[i]], real=True)
                self.final_fid.update(real_features, real=True)
            
            for j,data in tqdm(enumerate(self.fake_data_loader), total=len(self.fake_data_loader)):
                # if j>=20:
                #     break
                #data = self.fake_data[j]
                data = self.data_preprocessor(data)
                images = data['inputs']
                data_samples = data['data_samples']
                # Extract features from the real data
               
                fake_features = self.feature_extractor(images)
                for i in range(0, len(data_samples)):
                    meta_data = data_samples[i].metainfo
                    condition = self.fake_metadata_conditions.loc[
                        (self.fake_metadata_conditions['filename'] == meta_data['file_name']) 
                    ]['condition'].values[0]
                # Update the FID metric with the features
                    self.fid_dict[condition].update(fake_features[[i]], real=False)
                self.final_fid.update(fake_features, real=False)
            
            # Now compute the FID score for each condition 
            # and the total FID score over all conditions
            fid_scores = {}
            for condition in self.conditions:
                try:
                    fid_scores[condition] = self.fid_dict[condition].compute().item()
                except:
                    fid_scores[condition] = -1.0
            
            # Compute the total FID score
            fid_scores['total'] = self.final_fid.compute().item()
            
            # Save the FID scores to a json file
            path = self.config.SYN_DATASET_GEN.fid_results_path +\
            self.config.SYN_DATASET_GEN.dataset_path.split('/')[-2]+".json"
            
            json.dump(fid_scores, open(path, 'w'))
            
            
if __name__ == "__main__":
    config = omegaconf.OmegaConf.load("lang_data_synthesis/config.yaml")
    
    if torch.cuda.is_available():
        with torch.cuda.device(0):
            evaluate_fid = EvaluateSynthesisFID(config)
            evaluate_fid()
    else:
        evaluate_fid = EvaluateSynthesisFID(config)
        evaluate_fid()
            
            