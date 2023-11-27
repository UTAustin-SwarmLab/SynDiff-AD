import pandas as pd
from typing import *

def compute_accuracy(
    total_area_intersect:pd.Series,
    total_area_label:pd.Series):
    acc = total_area_intersect/total_area_label
    return acc


def compute_IOU(total_area_intersect, total_area_union):
    iou = total_area_intersect / total_area_union
    return iou

def compute_DICE(total_area_intersect, total_area_union):
    dice = 2 * total_area_intersect / (total_area_intersect + total_area_union)
    return dice

def compute_F1(total_area_intersect, total_area_pred_label, total_area_label):
    precision = total_area_intersect / total_area_pred_label
    recall = total_area_intersect / total_area_label
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1
