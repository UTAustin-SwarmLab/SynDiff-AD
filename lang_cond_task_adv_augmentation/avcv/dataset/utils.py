from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import immutabledict
import numpy as np
from skimage import segmentation


from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2
from waymo_open_dataset.v2.perception import segmentation as v2
from waymo_open_dataset.utils import camera_segmentation_utils
from PIL import Image
import io

LabelProtoOrComponent = Union[
    dataset_pb2.CameraSegmentationLabel, v2.CameraSegmentationLabelComponent
]
InstanceIDToGlobalIDMapping = (
    dataset_pb2.CameraSegmentationLabel.InstanceIDToGlobalIDMapping
)

def decode_multi_frame_panoptic_labels_from_segmentation_labels(
    segmentation_proto_list: Sequence[LabelProtoOrComponent],
    remap_to_global: bool = True,
    remap_to_sequential: bool = False,
    new_panoptic_label_divisor: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
  """Parses a set of panoptic labels with consistent instance ids from labels.

  This functions supports an arbitrary number of CameraSegmentationLabels,
  and can remap values both within the same and between different sequences.

  Both protos and components are supported by this function.

  Args:
    segmentation_proto_list: a sequence of CameraSegmentationLabel protos.
    remap_to_global: if true, will remap the instance ids using the
      instance_id_to_global_id_mapping such that they are consistent within each
      sequence, and that they are consecutive between all output labels.
    remap_to_sequential: whether to remap the instance IDs to sequential values.
      This is not recommended for eval.
    new_panoptic_label_divisor: if provided, the output panoptic label will be
      shifted to this new panoptic label divisor.

  Returns:
    A tuple containing
      panoptic_labels: a list of uint32 numpy arrays, containing the parsed
        panoptic labels. If any labels have instance_id_to_global_id_mappings,
        instances with these mappings will be mapped to the same value across
        frames.
      num_cameras_covered: a list of uint8 numpy arrays, containing the parsed
        arrays representing the number of cameras covered for each pixel. This
        is used to compute the weighted Segmentation and Tracking Quality (wSTQ)
        metric.
      is_tracked_masks: a list of uint8 numpy arrays, where a pixel is True if
        its instance is tracked over multiple frames.
      panoptic_label_divisor: the int32 divisor used to generate the panoptic
        labels.

  Raises:
    ValueError: if the output panoptic_label_divisor is lower than the maximum
      instance id.
  """
  # Use the existing panoptic_label_divisor if possible to maintain consistency
  # in the data and keep the panoptic labels more comprehensible.
  if new_panoptic_label_divisor is None:
    panoptic_label_divisor = segmentation_proto_list[0].panoptic_label_divisor
  else:
    panoptic_label_divisor = new_panoptic_label_divisor

  if remap_to_global:
    global_id_mapping = camera_segmentation_utils._remap_global_ids(
        segmentation_proto_list, remap_to_sequential=remap_to_sequential)
    if global_id_mapping:
      max_instance_id = max(
          [max([global_id for _, global_id in mapping.items()])
           for _, mapping in global_id_mapping.items()])
      if new_panoptic_label_divisor is None:
        panoptic_label_divisor = max(panoptic_label_divisor, max_instance_id)

  panoptic_labels = []
  num_cameras_covered = []
  is_tracked_masks = []
  for label in segmentation_proto_list:
    sequence = label.sequence_id
    panoptic_label = decode_single_panoptic_label_from_proto(label)
    semantic_label, instance_label = (
        camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
            panoptic_label, label.panoptic_label_divisor))
    camera_coverage = np.ones_like(instance_label, dtype=np.uint8)
    is_tracked_mask = np.zeros_like(instance_label, dtype=np.uint8)

    instance_label_copy = np.copy(instance_label)
    if remap_to_global:
      if isinstance(label, v2.CameraSegmentationLabelComponent):
        mapping_iter = camera_segmentation_utils._iterate_over_mapping(label)
      elif isinstance(label, dataset_pb2.CameraSegmentationLabel):
        mapping_iter = label.instance_id_to_global_id_mapping
      else:
        raise ValueError('Input label format not supported.')

      for mapping in mapping_iter:
        instance_mask = (instance_label == mapping.local_instance_id)
        is_tracked_mask[instance_mask] = mapping.is_tracked
        instance_label_copy[instance_mask] = global_id_mapping[sequence][
            mapping.global_instance_id]

    if np.amax(instance_label) >= panoptic_label_divisor:
      raise ValueError('A panoptic_label_divisor of '
                       f'{panoptic_label_divisor} is requested, but the '
                       'maximum instance id exceeds this.')

    if label.num_cameras_covered:
      camera_coverage = decode_png(label.num_cameras_covered)

    panoptic_labels.append(
        camera_segmentation_utils.encode_semantic_and_instance_labels_to_panoptic_label(
            semantic_label, instance_label_copy, panoptic_label_divisor))
    num_cameras_covered.append(camera_coverage)
    is_tracked_masks.append(is_tracked_mask)

  return (
      panoptic_labels,
      is_tracked_masks,
      num_cameras_covered,
      panoptic_label_divisor,
  )
  
  
def decode_single_panoptic_label_from_proto(
    segmentation_proto: LabelProtoOrComponent,
) -> np.ndarray:
  """Decodes a panoptic label from a CameraSegmentationLabel.

  Args:
    segmentation_proto: a CameraSegmentationLabel to be decoded.

  Returns:
    A 2D numpy array containing the per-pixel panoptic segmentation label.
  """
  return decode_png(segmentation_proto.panoptic_label)
  

def decode_png(input_data):
    """
    Decodes a PNG image to a PyTorch tensor.

    Args:
        input_data (bytes): The PNG image data in bytes.

    Returns:
        torch.Tensor: Decoded image as a tensor.
    """
    # Open the image from bytes data and convert to RGB (in case it's not)
    
    image = Image.open(io.BytesIO(input_data))
    # Convert the image to a PyTorch tensor
    return np.array(image, dtype=np.uint16)
