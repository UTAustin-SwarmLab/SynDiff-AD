import os

class GlobalConfig:
    """ base architecture configurations """
	# Data
    seq_len = 1 # input timesteps
    pred_len = 4 # future waypoints predicted
    tot_len = seq_len + pred_len

    root_dir = '/store/harsh/carla_data_neat/'
    train_towns = ["expert"]
    val_towns = ["experteval"]
    train_data, val_data = [], []
    for town in train_towns:
        train_data.append(os.path.join(root_dir, town))
        #train_data.append(os.path.join(root_dir, town+'_small'))
    for town in val_towns:
        val_data.append(os.path.join(root_dir, town))

    image_encoder_type = 'resnet34'

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras
    n_views = 3 # no. of camera views

    input_resolution = 256

    scale = 1 # image pre-processing, downscale factor

    num_class = 7
    classes = {
        0: [0, 0, 0],        # unlabeled
        1: [0, 0, 255],    # vehicle
        2: [128, 64, 128],   # road
        3: [255, 0, 0],     # red light
        4: [0, 255, 0],     # green light
        5: [0, 255, 255],     # pedestrian
        6: [157, 234,  50],    # road line
        # 6: [255, 255, 255], # sidewalk
    }
    converter = [
    0,    # unlabeled
    0,    # building
    0,    # fence
    0,    # other
    5,    # pedestrian
    0,    # pole
    6,    # road line
    2,    # road
    0,    # sidewalk
    0,    # vegetation
    1,    # vehicle
    0,    # wall
    0,    # traffic sign
    0,    # sky
    0,    # ground
    0,    # bridge
    0,    # rail track
    0,    # guard rail
    0,    # traffic light
    0,    # static
    0,    # dynamic
    0,    # water
    0,    # terrain
    3,    # red light
    3,    # yellow light
    4,    # green light
    0,    # stop sign
    6,    # stop line marking
    ]

    lr = 1e-4
    ls_seg = 1.0
    ls_depth = 10.0
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    # Conv Encoder
    vert_anchors = 8
    horz_anchors = 8
    anchors = vert_anchors * horz_anchors

	# GPT Encoder
    n_embd = 512 # default: 512
    block_exp = 4
    n_layer = 8 # default: 8
    n_head = 4 # default: 4
    n_scale = 4 # default: 4

    # Controller
    plan_points = 5 # length of grid sampled for locating waypoints
    plan_iters = 3 # recurrence count for locating waypoints
    turn_KP = 1.25 # default: 1.25
    turn_KI = 0.75 # default: 0.75
    turn_KD = 0.3 # default: 0.3
    turn_n = 40 # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40 # buffer size

    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.4 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
