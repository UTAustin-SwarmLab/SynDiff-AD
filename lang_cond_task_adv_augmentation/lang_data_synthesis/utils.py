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

# Define the ADE20K Palette
ADE_20K_PALETTE = dict(
{
    'wall': [120, 120, 120],
    'building': [180, 120, 120],
    'sky': [6, 230, 230],
    'floor': [80, 50, 50],
    'tree': [4, 200, 3],
    'ceiling': [120, 120, 80],
    'road': [140, 140, 140],
    'bed': [204, 5, 255],
    'windowpane': [230, 230, 230],
    'grass': [4, 250, 7],
    'cabinet': [224, 5, 255],
    'sidewalk': [235, 255, 7],
    'person': [150, 5, 61],
    'earth': [120, 120, 70],
    'door': [8, 255, 51],
    'table': [255, 6, 82],
    'mountain': [143, 255, 140],
    'plant': [204, 255, 4],
    'curtain': [255, 51, 7],
    'chair': [204, 70, 3],
    'car': [0, 102, 200],
    'water': [61, 230, 250],
    'painting': [255, 6, 51],
    'sofa': [11, 102, 255],
    'shelf': [255, 7, 71],
    'house': [255, 9, 224],
    'sea': [9, 7, 230],
    'mirror': [220, 220, 220],
    
    'rug': [255, 9, 92],
    'field': [112, 9, 255],
    'armchair': [8, 255, 214],
    'seat': [7, 255, 224],
    'fence': [255, 184, 6],
    'desk': [10, 255, 71],
    'rock': [255, 41, 10],
    'wardrobe': [7, 255, 255],
    'lamp': [224, 255, 8],
    'bathtub': [102, 8, 255],
    'railing': [255, 61, 6],
    'cushion': [255, 194, 7],
    'base': [255, 122, 8],
    'box': [0, 255, 20],
    'column': [255, 8, 41],
    'signboard': [255, 5, 153],
    'chest of drawers': [6, 51, 255],
    'counter': [235, 12, 255],
    'sand': [160, 150, 20],
    'sink': [0, 163, 255],
    'skyscraper': [140, 140, 140],
    'fireplace': [250, 10, 15],
    'refrigerator': [20, 255, 0],
    'grandstand': [31, 255, 0],
    'path': [255, 31, 0],
    'stairs': [255, 224, 0],
    'runway': [153, 255, 0],
    'case': [0, 0, 255],
    'pool table': [255, 71, 0],
    'pillow': [0, 235, 255],
    'screen door': [0, 173, 255],
    'stairway': [31, 0, 255],
    'river': [11, 200, 200],
    'bridge': [255, 82, 0],
    'bookcase': [0, 255, 245],
    'blind': [0, 61, 255],
    'coffee table': [0, 255, 112],
    'toilet': [0, 255, 133],
    'flower': [255, 0, 0],
    'book': [255, 163, 0],
    'hill': [255, 102, 0],
    'bench': [194, 255, 0],
    'countertop': [0, 143, 255],
    'stove': [51, 255, 0],
    'palm': [0, 82, 255],
    'kitchen island': [0, 255, 41],
    'computer': [0, 255, 173],
    'swivel chair': [10, 0, 255],
    'boat': [173, 255, 0],
    'bar': [0, 255, 153],
    'arcade machine': [255, 92, 0],
    'hovel': [255, 0, 255],
    'bus': [255, 0, 245],
    'towel': [255, 0, 102],
    'light': [255, 173, 0],
    'truck': [255, 0, 20],
    'tower': [255, 184, 184],
    'chandelier': [0, 31, 255],
    'awning': [0, 255, 61],
    'streetlight': [0, 71, 255],
    'booth': [255, 0, 204],
    'television receiver': [0, 255, 194],
    'airplane': [0, 255, 82],
    'dirt track': [0, 10, 255],
    'apparel': [0, 112, 255],
    'pole': [51, 0, 255],
    'land': [0, 194, 255],
    'bannister': [0, 122, 255],
    'escalator': [0, 255, 163],
    'ottoman': [255, 153, 0],
    'bottle': [0, 255, 10],
    'buffet': [255, 112, 0],
    'poster': [143, 255, 0],
    'stage': [82, 0, 255],
    'van': [163, 255, 0],
    'ship': [255, 235, 0],
    'fountain': [8, 184, 170],
    'conveyer belt': [133, 0, 255],
    'canopy': [0, 255, 92],
    'washer': [184, 0, 255],
    'plaything': [255, 0, 31],
    'swimming pool': [0, 184, 255],
    'stool': [0, 214, 255],
    'barrel': [255, 0, 112],
    'basket': [92, 255, 0],
    'waterfall': [0, 224, 255],
    'tent': [112, 224, 255],
    'bag': [70, 184, 160],
    'minibike': [163, 0, 255],
    'cradle': [153, 0, 255],
    'oven': [71, 255, 0],
    'ball': [255, 0, 163],
    'food': [255, 204, 0],
    'step': [255, 0, 143],
    'tank': [0, 255, 235],
    'trade name': [133, 255, 0],
    'microwave': [255, 0, 235],
    'pot': [245, 0, 255],
    'animal': [255, 0, 122],
    'bicycle': [255, 245, 0],
    'lake': [10, 190, 212],
    'dishwasher': [214, 255, 0],
    'screen': [0, 204, 255],
    'blanket': [20, 0, 255],
    'sculpture': [255, 255, 0],
    'hood': [0, 153, 255],
    'sconce': [0, 41, 255],
    'vase': [0, 255, 204],
    'traffic light': [41, 0, 255],
    'tray': [41, 255, 0],
    'ashcan': [173, 0, 255],
    'fan': [0, 245, 255],
    'pier': [71, 0, 255],
    'crt screen': [122, 0, 255],
    'plate': [0, 255, 184],
    'monitor': [0, 92, 255],
    'bulletin board': [184, 255, 0],
    'shower': [0, 133, 255],
    'radiator': [255, 214, 0],
    'glass': [25, 194, 194],
    'clock': [102, 255, 0],
    'flag': [92, 0, 255]
}
)

# Define the COCO Pallette
COCO_PALETTE = dict(
{
    'person': [220, 20, 60],
    'bicycle': [119, 11, 32],
    'car': [0, 0, 142],
    'motorcycle': [0, 0, 230],
    'airplane': [106, 0, 228],
    'bus': [0, 60, 100],
    'train': [0, 80, 100],
    'truck': [0, 0, 70],
    'boat': [0, 0, 192],
    'traffic light': [250, 170, 30],
    'fire hydrant': [100, 170, 30],
    'stop sign': [220, 220, 0],
    'parking meter': [175, 116, 175],
    'bench': [250, 0, 30],
    'bird': [165, 42, 42],
    'cat': [255, 77, 255],
    'dog': [0, 226, 252],
    'horse': [182, 182, 255],
    'sheep': [0, 82, 0],
    'cow': [120, 166, 157],
    'elephant': [110, 76, 0],
    'bear': [174, 57, 255],
    'zebra': [199, 100, 0],
    'giraffe': [72, 0, 118],
    'backpack': [255, 179, 240],
    'umbrella': [0, 125, 92],
    'handbag': [209, 0, 151],
    'tie': [188, 208, 182],
    'suitcase': [0, 220, 176],
    'frisbee': [255, 99, 164],
    'skis': [92, 0, 73],
    'snowboard': [133, 129, 255],
    'sports ball': [78, 180, 255],
    'kite': [0, 228, 0],
    'baseball bat': [174, 255, 243],
    'baseball glove': [45, 89, 255],
    'skateboard': [134, 134, 103],
    'surfboard': [145, 148, 174],
    'tennis racket': [255, 208, 186],
    'bottle': [197, 226, 255],
    'wine glass': [171, 134, 1],
    'cup': [109, 63, 54],
    'fork': [207, 138, 255],
    'knife': [151, 0, 95],
    'spoon': [9, 80, 61],
    'bowl': [84, 105, 51],
    'banana': [74, 65, 105],
    'apple': [166, 196, 102],
    'sandwich': [208, 195, 210],
    'orange': [255, 109, 65],
    'broccoli': [0, 143, 149],
    'carrot': [179, 0, 194],
    'hot dog': [209, 99, 106],
    'pizza': [5, 121, 0],
    'donut': [227, 255, 205],
    'cake': [147, 186, 208],
    'chair': [153, 69, 1],
    'couch': [3, 95, 161],
    'potted plant': [163, 255, 0],
    'bed': [119, 0, 170],
    'dining table': [0, 182, 199],
    'toilet': [0, 165, 120],
    'tv': [183, 130, 88],
    'laptop': [95, 32, 0],
    'mouse': [130, 114, 135],
    'remote': [110, 129, 133],
    'keyboard': [166, 74, 118],
    'cell phone': [219, 142, 185],
    'microwave': [79, 210, 114],
    'oven': [178, 90, 62],
    'toaster': [65, 70, 15],
    'sink': [127, 167, 115],
    'refrigerator': [59, 105, 106],
    'book': [142, 108, 45],
    'clock': [196, 172, 0],
    'vase': [95, 54, 80],
    'scissors': [128, 76, 255],
    'teddy bear': [201, 57, 1],
    'hair drier': [246, 0, 122],
    'toothbrush': [191, 162, 208],
    'banner': [255, 255, 128],
    'blanket': [147, 211, 203],
    'bridge': [150, 100, 100],
    'cardboard': [168, 171, 172],
    'counter': [146, 112, 198],
    'curtain': [210, 170, 100],
    'door-stuff': [92, 136, 89],
    'floor-wood': [218, 88, 184],
    'flower': [241, 129, 0],
    'fruit': [217, 17, 255],
    'gravel': [124, 74, 181],
    'house': [70, 70, 70],
    'light': [255, 228, 255],
    'mirror-stuff': [154, 208, 0],
    'net': [193, 0, 92],
    'pillow': [76, 91, 113],
    'towel': [225, 199, 255],
    'wall-brick': [137, 54, 74],
    'wall-stone': [135, 158, 223],
    'wall-tile': [7, 246, 231],
    'wall-wood': [107, 255, 200],
    'water-other': [58, 41, 149],
    'window-blind': [183, 121, 142],
    'window-other': [255, 73, 97],
    'tree-merged': [107, 142, 35],
    'fence-merged': [190, 153, 153],
    'ceiling-merged': [146, 139, 141],
    'sky-other-merged': [70, 130, 180],
    'cabinet-merged': [134, 199, 156],
    'table-merged': [209, 226, 140],
    'floor-other-merged': [96, 36, 108],
    'pavement-merged': [96, 96, 96],
    'mountain-merged': [64, 170, 64],
    'grass-merged': [152, 251, 152],
    'dirt-merged': [208, 229, 228],
    'paper-merged': [206, 186, 171],
    'food-other-merged': [152, 161, 64],
    'building-other-merged': [116, 112, 0],
    'rock-merged': [0, 114, 143],
    'wall-other-merged': [102, 102, 156],
    'rug-merged': [250, 141, 255]   
}
)