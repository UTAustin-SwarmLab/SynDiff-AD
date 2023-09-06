
from waymo_open_dataset.protos import camera_segmentation_metrics_pb2 as metrics_pb2
from waymo_open_dataset.protos import camera_segmentation_submission_pb2 as submission_pb2
from waymo_open_dataset.wdl_limited.camera_segmentation import camera_segmentation_metrics
from waymo_open_dataset.utils import camera_segmentation_utils

from waymo_open_data_parser.parser import *

from matplotlib import pyplot as plt
import gradio as gr

# TODO: set up the gradio display
def set_gradio_display():
  block = gr.Blocks().queue()
  with block:
      with gr.Row():
          gr.Markdown("## Visualize Waymo Open Dataset Camera Segmentation Labels")
      with gr.Row():
          with gr.Column():
              run_button = gr.Button(label="Run")
              with gr.Accordion("Advanced options", open=False):
                  #TODO: we need the maximum number of images in the dataset for visualization
                  image_number = gr.Slider(label="Image Resolution", minimum=0, maximum=10000, value=512, step=1)
                
          with gr.Column():
              result_gallery = gr.Image (label='Camera image', show_label=False, elem_id="gallery").style(grid=2, height='auto')

      ips = [image_number]
      run_button.click(fn=call_display, inputs=ips, outputs=[result_gallery])


  block.launch(server_name='0.0.0.0')


def _pad_to_common_shape(label):
  return np.pad(label, [[1280 - label.shape[0], 0], [0, 0], [0, 0]])


def visualize(config: omegaconf, 
              instance_labels_multiframe: List[open_dataset.CameraSegmentationLabel],
              semantic_labels_multiframe: List[open_dataset.CameraSegmentationLabel]) -> None:
  # TODO: Add gradio visualization 
  # Pad labels to a common size so that they can be concatenated.

  instance_labels = [[_pad_to_common_shape(label) for label in instance_labels] 
                     for instance_labels in instance_labels_multiframe]
  semantic_labels = [[_pad_to_common_shape(label) for label in semantic_labels]
                      for semantic_labels in semantic_labels_multiframe]
  instance_labels = [np.concatenate(label, axis=1) for label in instance_labels]
  semantic_labels = [np.concatenate(label, axis=1) for label in semantic_labels]

  instance_label_concat = np.concatenate(instance_labels, axis=0)
  semantic_label_concat = np.concatenate(semantic_labels, axis=0)
  panoptic_label_rgb = camera_segmentation_utils.panoptic_label_to_rgb(
      semantic_label_concat, instance_label_concat)
      

  plt.figure(figsize=(64, 60))
  plt.imshow(panoptic_label_rgb)
  plt.grid(False)
  plt.axis('off')
  plt.show()
     