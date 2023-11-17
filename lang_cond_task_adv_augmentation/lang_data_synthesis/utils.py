from typing import *
import numpy as np
import datetime
import csv
from pathlib import Path
import pandas as pd
import torch

def convert_pallette_segment(
        metadata:dict,
        obj_mask:np.ndarray,
        seg_mask_metadata:dict,
        extra_mapping:dict={},
        background_included = False
        ):
    '''
    Returns an image with the dataset specific pallete converted to 
    to a uniform segmentation mask.

    Args:
        metadata: The metadata of the dataset inclding class name and 
        palletes we want to convert to. By default we use the ADE20K pallette.

        seg_mask: The maks with object classes annotated 
        
        seg_mask_metadata: The metadata of the segmentation mask which includes the object
        clases
        extra_mapping: A dictionary with extra mappings to add to 
        consider for semantic segmentation
        increment:
    '''

    object_classes_seg = seg_mask_metadata['object_classes'] #List and ID of object classes
    if not background_included:
        pallete = np.array([[255,255,255]]+metadata['pallete'])
        metadata['object_classes'] = tuple(['-']+list(metadata['object_classes']))
    else:
        pallete = np.array(metadata['pallete'])

    object_classes_inverse_mapping = inverse_mapping(metadata['object_classes'])
    classes_to_convert = set(obj_mask.flatten().tolist())

    mapping = {}
    for c in classes_to_convert:
        if c ==0:
            mapping[c] = 0
        else:
            class_name = object_classes_seg[c]
            
            idx = object_classes_inverse_mapping.get(class_name, None)
            # Map all the objects that the model doesnt know as background
            if idx is not None:
                mapping[c] = idx
            else:
                class_name_ = extra_mapping.get(class_name, None)
                if class_name_ is not None:
                    mapping[c] = object_classes_inverse_mapping[class_name_]
                else:
                    #Try running through all the keys
                    classes = str.split(class_name, ', ')
                    for cla in classes:
                        idx = object_classes_inverse_mapping.get(cla, None)
                        if idx is not None:
                            mapping[c] = idx
                            break
                        else:
                            mapping[c] = 0
                    # for k in object_classes_inverse_mapping.keys():
                    #     if k in classes:
                    #         mapping[c] = object_classes_inverse_mapping[k]
                    #         break
                    #     else:
                    #         mapping[c] = 0
    
    seg_mask = np.zeros(tuple(list(obj_mask.shape) + [3]))
    mapped_object_mask = np.vectorize(mapping.get)(obj_mask)
    seg_mask = pallete[mapped_object_mask]
    object_keys = []
    for _,v in mapping.items():
        object_keys.append(metadata['object_classes'][v])
    return seg_mask, object_keys

def inverse_mapping(classes:List):
    '''
    Returns a dictionary with the inverse mapping of the classes
    '''
    return {v:k for k,v in enumerate(classes)}


def get_file_or_dir_with_datetime(base_name, ext=".",
                                  inc_current_time=True):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if inc_current_time:
        return f"{base_name}_{current_time}{ext}"
    else:
        return f"{base_name}{ext}"


def write_to_csv_from_dict(
    dict_data: dict, 
    csv_file_path: str, 
    file_name="data.csv"
):
    """Write dictionary to csv file."""
    csv_file_name = Path(csv_file_path) / file_name

    with open(csv_file_name, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=dict_data.keys())

        if not csv_file_name.exists():
            # If file does not exist, write header
            writer.writeheader()
        writer.writerow(dict_data)

def get_df_from_csv(csv_file_path: str, file_name="data.csv", header=None):
    csv_file_name = Path(csv_file_path) / file_name
    if not csv_file_name.exists():
        return None
    else:
        # If the file exists, read the CSV into a DataFrame

        # Add the first column as the row due to problem with the csv file

        df = pd.read_csv(csv_file_name)
        if header is not None and header.keys() != df.columns.tolist():
            df.loc[-1] = df.columns.tolist()
            df.index = df.index + 1
            df = df.sort_index()

        if header is not None:
            df.columns = header.keys()
            for col in header:
                if col in df.columns:
                    df[col] = df[col].astype(header[col])

    return df

def row_in_csv(
    df: pd.DataFrame, 
    dict_data: Dict[str, Any],
    keys_to_compare: List[str]
):
    if df is None:
        return False

    # Define the keys you want to compare

    # Check if the data already exists in the DataFrame for the selected keys
    exists = df[keys_to_compare].eq(pd.Series(dict_data)[keys_to_compare]).all(axis=1).any()
    
    return exists


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor
    
def to_cpu(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    else:
        return tensor