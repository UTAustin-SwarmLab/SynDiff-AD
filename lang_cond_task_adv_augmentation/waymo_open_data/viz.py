
# from waymo_open_dataset.protos import camera_segmentation_metrics_pb2 as metrics_pb2
# from waymo_open_dataset.protos import camera_segmentation_submission_pb2 as submission_pb2
# from waymo_open_dataset.wdl_limited.camera_segmentation import camera_segmentation_metrics
from waymo_open_dataset.utils import camera_segmentation_utils
from waymo_open_dataset import v2
from parser import *

from matplotlib import pyplot as plt
import gradio as gr
import omegaconf
from typing import *
import numpy as np
from waymo_open_data.parser import *

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
              instance_labels_multiframe: List[v2.CameraSegmentationLabel],
              semantic_labels_multiframe: List[v2.CameraSegmentationLabel],
              camera_images: List[np.ndarray]) -> None:
    # TODO: Add gradio visualization 
    # Pad labels to a common size so that they can be concatenated.
    for j,(instance_labels, semantic_labels, cam_images) in enumerate(zip(instance_labels_multiframe, 
                                                                     semantic_labels_multiframe, camera_images)):
        ilabels = [_pad_to_common_shape(label) for label in instance_labels]
        slabels = [_pad_to_common_shape(label) for label in semantic_labels]
        cimage = [_pad_to_common_shape(image) for image in cam_images]
                        
        # ilabels = [np.concatenate(label, axis=0) for label in ilabels]
        # slabels = [np.concatenate(label, axis=0) for label in slabels]

        instance_label_concat = np.concatenate(ilabels, axis=1)
        semantic_label_concat = np.concatenate(slabels, axis=1)
        camera_image_concat = np.concatenate(cimage, axis=1)
        panoptic_label_rgb = camera_segmentation_utils.panoptic_label_to_rgb(
            semantic_label_concat, instance_label_concat)
            
        cam_with_panopitc = np.concatenate(
            [camera_image_concat, panoptic_label_rgb],
            axis=0
        )

        plt.figure(figsize=(64, 60))
        plt.imshow(panoptic_label_rgb)
        plt.grid(False)
        plt.axis('off')
        plt.show()
        plt.savefig(os.path.join(config.VIZ_DIR,'panoptic_label_rgb{}.png'.format(j)))

        plt.figure(figsize=(64, 60))
        plt.imshow(cam_with_panopitc)
        plt.grid(False)
        plt.axis('off')
        plt.show()
        plt.savefig(os.path.join(config.VIZ_DIR,'cam_with_panoptic_rgb{}.png'.format(j)))

if __name__ == '__main__':
    config = omegaconf.OmegaConf.load('config.yaml')
    config.TRAIN_DIR = os.path.join(os.getcwd(), config.TRAIN_DIR)
    config.VIZ_DIR = os.path.join(os.getcwd(), config.VIZ_DIR)
    if not os.path.exists(config.VIZ_DIR):
        os.makedirs(config.VIZ_DIR)
        
    context = "1005081002024129653_5313_150_5333_150"
    context_frame = [1510593618340205, 1510593607540181] # None

    
    frames_with_seg, camera_images = load_data_set_parquet(config, context, 
                                                           context_frames=context_frame)
    semantic_labels_multiframe, instance_labels_multiframe, panoptic_labels = read_semantic_labels(config,frames_with_seg)
    camera_images_frame = read_camera_images(config, camera_images)


    visualize(config, instance_labels_multiframe, semantic_labels_multiframe, camera_images_frame)
