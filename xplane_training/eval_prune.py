# python3 xplane/eval_network.py -expr_name ad -network_type td -num_examples 10000

import os
import sys

import diffcp
import numpy as np
import argparse
import torch
from IPython import embed
DEVICE_NAME = "pulkit"
frac = "0.1"
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

# from xplane_training.utils import MPCCost
from xplane_training.xplane_traning.planners.xplane_mpc_double_integrator import track_traj, NX, TARGET_SPEED, T
from xplane_training.xplane_traning.planners.xplane_mpc import MPC_LEN, NU, NX, TARGET_SPEED, solve_mpc_tracking
from xplane_training.xplane_traning.perception_models import PerceptionWaypointModel
from xplane_training.xplane_traning.soft_vae import SoftIntroVAE
# NX = 
from itertools import islice

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

from xplane_training.taxinet_dataloader_prune import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    taxinet_prepare_dataloader,
    ExprType,
)
from xplane_training.utils import ENDC, OKBLUE, OKGREEN, OKRED, OKYELLOW, SEED

torch.manual_seed(SEED)
Z_DIM = 128
PI = 3.1415927
PATH_LENGTH = 1
HISTORY_LENGTH = 4

BASE_NPZ_PATH = "./eval_npz_data"

# base_waypoint_path = "xplane/scratch/taxinet_Waypoint4_Z128_soft_vae_train"
# base_waypoint_path = "/home/somi-admin/Desktop/task_driven_training/xplane/trained_models/taxinet_Waypoint4_Z128_svae_all_data"
expr_name_to_save_weights_path = {
    "orig": f"/home/pulkit/datasets_xplane/".replace("pulkit", DEVICE_NAME)+frac+"/taxinet_wp/best_model_wp.pt",
    # "da": f"{base_waypoint_path}/afternoon_data_addition/best_model.pt",
    # "curl": f"{base_waypoint_path}/afternoon_curl/best_model.pt",
    # "td": f"{base_waypoint_path}/afternoon_task_driven/best_model.pt",
}

# vae_save_path = f"xplane/scratch/taxinet_VAE_Z{Z_DIM}_train/afternoon/best_model_45.pt"
# vae_save_path = f"/home/somi-admin/Desktop/task_driven_training/xplane/trained_models/taxinet_VAE_Z{Z_DIM}_all_data/afternoon/best_model_55.pt"
vae_save_path = f"/home/pulkit/datasets_xplane/".replace("pulkit", DEVICE_NAME)+frac+"/trained_models_svae/taxinet_VAE_Z128_all_data/afternoon/best_model_svae.pt"

def orig_to_network_wp(orig: torch.Tensor, train_dataset: torch.utils.data.Dataset) -> torch.Tensor:
    pred_wp = orig.reshape(PATH_LENGTH, 2)
    pred_wp = train_dataset.inverse_transform_torch(pred_wp)
    mpc_pred_wp = torch.zeros((NX))
    mpc_pred_wp[1] = pred_wp[:, 0]  # Distance to centerline
    mpc_pred_wp[2] = pred_wp[:, 1] * PI / 180.0  # Heading Error
    mpc_pred_wp[3] = TARGET_SPEED  # Mantain constant velocity.
    return mpc_pred_wp


def costs_struct_to_np_array(mpc_costs, cost_np: np.ndarray) -> np.ndarray:
    # mpc_costs.total = min(2000.0, mpc_costs.total)
    # mpc_costs.tracking = min(1500.0, mpc_costs.tracking)
    # mpc_costs.control = min(500.0, mpc_costs.control)
    # mpc_costs.goal = min(0.0, mpc_costs.goal)

    cost_np[0] = mpc_costs.total
    cost_np[1] = mpc_costs.tracking
    cost_np[2] = mpc_costs.control
    cost_np[3] = mpc_costs.goal

    return cost_np


def eval_network_for_network_on_data(args, model, data_dir, dataloader_params, train_dataset):
    # assert len(args.data_type) == 1, "Only 1 datatype at a time supported currently."

    # DATALOADERS
    val_dataset, val_loader = taxinet_prepare_dataloader(
        data_dir,
        condition_str,
        "val",
        dataloader_params,
        HISTORY_LENGTH,
        train_dataset.scaler,
        args.data_type,
    )
    IMGNS = val_dataset.all_images
    # wpt_error = np.zeros((args.num_examples, 1))
    wpt_error = np.zeros((len(IMGNS), 1))
    # print(wpt_error.shape)
    # print("images: ", IMGNS)
    dict_wpt_err = {}
    dict_wpt = {}
    # samples = np.zeros((args.num_examples, Z_DIM))
    from tqdm import tqdm
    for i, data in tqdm(enumerate(val_loader)):
        # print(val_dataset)
        if i >= args.num_examples:
            break

        scene, history, waypoint, _ = data
        scene = scene.to(device)
        history = history.to(device)
        waypoint = waypoint.to(device)

        # with torch.autograd.set_detect_anomaly(True):
        sample, _ = model.vae.encode(scene)
        # print(sample.shape)
        # samples[i] = sample.clone().detach().numpy()
        pred_wp, recon_scene = model.adversarial_forward(sample, history=history)
        # from IPython import embed; embed()

        wpt_error = ((waypoint - pred_wp) ** 2).mean(1).detach().cpu().numpy()
        J = 0
        for j in range(wpt_error.shape[0]):
            # print(IMGNS[args.batch_size*i+j])
            dict_wpt_err[IMGNS[args.batch_size*i+j]] = wpt_error[j]
            if not "night" in IMGNS[args.batch_size*i+j]:
                dict_wpt[IMGNS[args.batch_size*i+j]] = (pred_wp).mean(1).detach().cpu().numpy()[j]
            J = j
        # print((i+1)*(J+1))
    return dict_wpt_err, dict_wpt
            # mpc_pred_wp = orig_to_network_wp(pred_wp, train_dataset)
            # mpc_label_wp = orig_to_network_wp(waypoint, train_dataset)

            # try:
            #     network_mpc_costs, network_opt_state, network_opt_u = solve_mpc_tracking(mpc_pred_wp, mpc_label_wp)

            #     # We will give label waypoint as the MPC problem input
            #     label_mpc_costs, label_opt_state, label_opt_u = solve_mpc_tracking(mpc_label_wp, mpc_label_wp)
            # except diffcp.cone_program.SolverError:
            #     print(OKBLUE + "Unbounded Infeasible" + ENDC)
            #     # mpc_costs = MPCCost(total=0.0, tracking=0.0, control=0.0, goal=0.0)
            #     continue

            # network_total_cost[i] = costs_struct_to_np_array(network_mpc_costs, network_total_cost[i])
            # label_total_cost[i] = costs_struct_to_np_array(label_mpc_costs, label_total_cost[i])

    #         network_states[i], network_controls[i] = (
    #             network_opt_state.detach().numpy().T,
    #             network_opt_u.detach().numpy().T,
    #         )
    #         label_states[i], label_controls[i] = label_opt_state.detach().numpy().T, label_opt_u.detach().numpy().T

    #     if args.debug_mode:
    #         print(
    #             f"Example {i}, Network Total Cost: {network_mpc_costs.total:.2f}, Label Total Cost: {label_mpc_costs.total:.2f}"
    #             # f" Control: {mpc_costs.control:.2f}, Goal: {mpc_costs.goal:.2f}"
    #         )

    # print(OKRED + f"Network Avg Cost: {np.mean(network_total_cost[:, 0])}" + ENDC)
    # print(OKRED + f"Label Avg Cost: {np.mean(label_total_cost[:, 0])}" + ENDC)
    # print(OKGREEN + f"Avg Waypoint Error : {np.mean(wpt_error)}" + ENDC)

    # np.savez(
    #     f"{BASE_NPZ_PATH}/{args.expr_name}_{args.network_type}_{args.data_type[0].name}_data_{network_total_cost.shape[0]}",
    #     network_total_cost,  # 0
    #     label_total_cost,  # 1
    #     wpt_error,  # 2
    #     network_states,  # 3
    #     network_controls,  # 4
    #     label_states,  # 5
    #     label_controls,  # 6
    # )
    # np.savez(f"{BASE_NPZ_PATH}/{args.expr_name}_{args.data_type[0].name}_data_{network_total_cost.shape[0]}", samples)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda:1")

    base_path = "./xplane/"
    num_ex = len([x for x in os.listdir("/home/pulkit/imgs/".replace("pulkit", DEVICE_NAME)) if x.endswith(".jpg")])
    parser = argparse.ArgumentParser(description="Args for Evaluating Networks for Robust Training")
    parser.add_argument("-expr_name", type=str, default="ok")
    parser.add_argument("-network_type", "-nt", type=str, default="orig")
    parser.add_argument("-num_examples", "-n", type=int, default =num_ex)
    parser.add_argument("-data_type", type=str, nargs="+", required=False)
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--debug_mode", dest="debug_mode", action="store_true")
    parser.set_defaults(debug_mode=False)

    args = parser.parse_args()
    print(OKYELLOW + f"Running {args.expr_name} experiment" + ENDC)
    assert (
        args.network_type in expr_name_to_save_weights_path.keys()
    ), f"Network Type needs to be one of: {expr_name_to_save_weights_path.keys()}"

    condition_str = "afternoon"

    vae_model = SoftIntroVAE(
        cdim=3,
        zdim=Z_DIM,
        channels=[64, 128, 256, 512, 512],
        image_size=IMAGE_HEIGHT,
    ).to(device)

    model = PerceptionWaypointModel(
        vae_model,
        input_size=Z_DIM + 2*HISTORY_LENGTH,
        output_size=PATH_LENGTH * 2,
        device=device,
        vae_save_path=vae_save_path,
        waypoint_len=PATH_LENGTH,
        small=True,
        check_frozen=False
    ).to(device)
    # print(model)
    args.waypoint_path = expr_name_to_save_weights_path[args.network_type]
    chkpt = torch.load(args.waypoint_path, map_location=torch.device(device))
    print(f"Using weights {args.waypoint_path}")
    model.load_state_dict(chkpt)
    model.eval()

    # data_dir = f"/home/somi-admin/Desktop/task_driven_training/xplane/xplane_data/data_original/{condition_str}/"
    data_dir = "/home/pulkit/triplets/".replace("pulkit", DEVICE_NAME)
    
    all_images = [x for x in os.listdir(data_dir)]
    print("Running over " + str(len(all_images)) + " images.")
    night_images = [x for x in os.listdir(data_dir) if "night" in x]
    day_images = [x for x in os.listdir(data_dir) if "day" in x]
    real_images = [x for x in all_images if x not in night_images and x not in day_images]
    dataloader_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 24,
        "drop_last": False,
        "pin_memory": False,
    }

    # instantiate the model and freeze all but penultimate layers
    # from xplane_training.taxinet_dataloader import *
    import xplane_training.taxinet_dataloader
    data_dir_main = "/home/pulkit/datasets_xplane/".replace("pulkit", DEVICE_NAME)+frac+"/xplane_dataset/"
    train_dataset, train_loader = xplane_training.taxinet_dataloader.taxinet_prepare_dataloader(
        data_dir_main,   
        condition_str,
        "train",
        dataloader_params,
        HISTORY_LENGTH,
        None,
        frac,
        [ExprType.standard],
        )
    print("SIZE OF TRAINING DATA: ", len(train_dataset))
    # if args.data_type is None:
    #     for dt in [
    #         ExprType.orig,
    #         ExprType.morning,
    #         ExprType.night,
    #         ExprType.task_driven,
    #         # ExprType.overcast,
    #     ]:
    #         args.data_type = [dt]
    #         eval_network_for_network_on_data(args, model, data_dir, dataloader_params, train_dataset)
    # else:
    #     args.data_type = [ExprType[s] for s in args.data_type]
    #     eval_network_for_network_on_data(args, model, data_dir, dataloader_params, train_dataset)

    WP_DATA, WP_ABS_DATA = eval_network_for_network_on_data(args, model, data_dir, dataloader_params, train_dataset)
    wp_data_score_fix = {}
    d2 = {}
    s = 100000000000
    s_i =  ""
    for images in real_images:
        if images in WP_DATA.keys() and (((images.replace("_real.png","_night.jpg") in WP_DATA.keys()) and (images.replace("_real.jpg","_day.jpg")  in WP_DATA.keys())) or ((images.replace("_real.png","_night.png") in night_images) and (images.replace("_real.png","_day.png") in day_images))):
            waypoint_real = WP_DATA[images]
            waypoint_resynthesized = WP_DATA[images.replace("_real","_day")]
            error_synthetic = WP_DATA[images.replace("_real","_night")]
            error_real = WP_DATA[images]
            error_resynthesized = WP_DATA[images.replace("_real","_day")]
            
            # p_change_r2d = abs(error_real - error_resynthesized)
            # p_change_r2s = abs(error_real - error_synthetic)
            # p_change_s2d = abs(error_synthetic - error_resynthesized)
            # wp_data_score_fix[images] = -p_change_r2s + 4*p_change_r2d
            wp_data_score_fix[images] = (-error_synthetic + 500*abs(waypoint_real - waypoint_resynthesized) + 2000*error_real)
            # wp_data_score_fix[images] = ((3*abs(error_real - error_resynthesized)/error_real - 0.1*abs(error_real - error_synthetic)/error_real))
            # d2[images] = [error_real, error_synthetic, error_resynthesized, ]
    
    wp_data_score_sort = {k: v for k, v in sorted(wp_data_score_fix.items(), key=lambda item: item[1])}
    import csv
    print("Best Images: ", take(5, wp_data_score_sort.items()))
    with open('/home/pulkit/wp_data_score.csv'.replace("pulkit", DEVICE_NAME), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in wp_data_score_sort.items():
            writer.writerow([key, value])
    with open('/home/pulkit/consoldiated.csv'.replace("pulkit", DEVICE_NAME), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in d2.items():
            writer.writerow([key, value])

    print("rank of good triplets")
    print("aft 16 1895", list(wp_data_score_sort).index('MWH_Runway04_afternoon_clear_16_1895_real.png'))
    print("aft 15 923", list(wp_data_score_sort).index('MWH_Runway04_afternoon_clear_15_923_real.png'))
    print("aft 16 996", list(wp_data_score_sort).index('MWH_Runway04_afternoon_clear_16_996_real.png'))
    print("aft 1 1516", list(wp_data_score_sort).index('MWH_Runway04_afternoon_clear_1_1516_real.png'))
    print("mng 12 317", list(wp_data_score_sort).index('MWH_Runway04_morning_clear_12_317_real.png'))
    print("mng 5 890", list(wp_data_score_sort).index('MWH_Runway04_morning_clear_5_890_real.png'))
    print("mng 2 109", list(wp_data_score_sort).index('MWH_Runway04_morning_clear_2_109_real.png'))
    print("mng 17 1377", list(wp_data_score_sort).index('MWH_Runway04_morning_clear_17_1377_real.png'))

    # print("SEARCHING FOR BEST GAMMA.....")
    # for gamma in [0.01,0.02, 0.04, 0.08, 0.16, 0.24, 0.36, 0.50, 0.72, 0.81, 0.9, 1.2, 1.4, 1.65, 1.95, 2.5, 2.6, 2.8, 3.2, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
    #     for images in real_images:
    #         # print(images)
    #         if ((images.replace("_real.jpg","_night.jpg") in WP_DATA.keys()) and (images.replace("_real.jpg","_day.jpg") in WP_DATA.keys())) or ((images.replace("_real.png","_night.png") in night_images) and (images.replace("_real.png","_day.png") in day_images)):
    #             error_real = WP_DATA[images]
    #             error_synthetic = WP_DATA[images.replace("_real","_night")]
    #             error_resynthesized = WP_DATA[images.replace("_real","_day")]
                
    #             p_change_r2d = abs(error_real - error_resynthesized)/error_real
    #             p_change_r2s = (error_real - error_synthetic)/error_real
    #             p_change_s2d = abs(error_synthetic - error_resynthesized)/error_resynthesized
    #             wp_data_score_fix[images] = (p_change_r2d + gamma*(p_change_r2s))
    #     wp_data_score_sort = {k: v for k, v in sorted(wp_data_score_fix.items(), key=lambda item: item[1])}
    #     print("gamma: ", gamma)
    #     print(take(3, wp_data_score_sort.items()))