from typing import *
import numpy as np

def convert_pallette_segment(
        metadata:dict,
        obj_mask:np.ndarray,
        seg_mask_metadata:dict,
        extra_mapping:dict=None,
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
    pallete = np.array([[255,255,255]]+metadata['pallete'])
    metadata['object_classes'] = tuple(['-']+list(metadata['object_classes']))
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