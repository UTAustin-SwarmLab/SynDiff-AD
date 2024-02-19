
import numpy as np
from swarm_visualizer.boxplot import  plot_paired_boxplot
from swarm_visualizer.utility.general_utils import save_fig, set_plot_properties
from matplotlib import pyplot as plt
import seaborn as sns
from utils import write_to_csv_from_dict
from copy import deepcopy
from tqdm import tqdm
import torch.utils.data.dataloader as DataLoader
from bdd100k.data_loader import BDD100KDataset

from waymo_open_data import WaymoDataset
import omegaconf
from dataset import collate_fn as collator
import functools
from argparse import ArgumentParser

def write_metadata(dataset, class_meta_file_path):

    CLASSES = dataset.CLASSES
    keys = list(dataset.METADATA[0].keys())
    keys += CLASSES
    data_dict= {k:k for k in keys}
    if 'condition' in data_dict:
        del data_dict['condition']
    write_to_csv_from_dict(
                dict_data=data_dict , 
                csv_file_path= class_meta_file_path,
                file_name=""
        )

    collate_fn = functools.partial(
                    collator, 
                    segmentation=dataset.segmentation,
                    image_meta_data=dataset.image_meta_data
                )

    dataloader = DataLoader.DataLoader(
                dataset=dataset,
                batch_size=8,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4
            )

    for _,data in tqdm(enumerate(dataloader), total=len(dataloader)):
        
        imgs = data[0]
        masks = data[3]
        metas = data[4]
        
        for _, mask, meta in zip(imgs, masks, metas):
            object_set = set(mask.flatten().tolist())
            object_classes = set([CLASSES[object] for object in object_set])
            data_dict = deepcopy(meta)
            
            if 'condition' in data_dict:
                del data_dict['condition']
                
            for sem in CLASSES:
                if sem in object_classes:
                    data_dict[sem] = 1
                else:
                    data_dict[sem] = 0
                    
            write_to_csv_from_dict(
                dict_data=data_dict ,
                csv_file_path= class_meta_file_path,
                file_name=""
            )
  
def parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="waymo",
        choices=["waymo", "bdd"],
        help="dataset"
    )

    return parser.parse_args()     
if __name__=="__main__":
    
    args = parser()
    if args.experiment == "waymo":
        config_file_name = 'lang_data_synthesis/waymo_config.yaml'
        config = omegaconf.OmegaConf.load(config_file_name)
        dataset = WaymoDataset(config.IMAGE.WAYMO, image_meta_data=True)

        class_meta_file_path = 'waymo_open_data/waymo_class_metadata.csv'
    elif args.experiment == "bdd":
        config_file_name = 'lang_data_synthesis/bdd_config.yaml'
        config = omegaconf.OmegaConf.load(config_file_name)
        dataset = BDD100KDataset(config.IMAGE.BDD, image_meta_data=True)

        class_meta_file_path = 'bdd100k/bdd_class_metadata.csv'
        
    write_metadata(dataset, class_meta_file_path)