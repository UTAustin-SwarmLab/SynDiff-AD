# python3 xplane/eval_network.py -expr_name ad -network_type td -num_examples 10000

import os
import sys

import diffcp
import numpy as np
import argparse
import torch
from IPython import embed

device_name = "pulkit"
DEVICE_NAME = device_name
split = "0.4"
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

from xplane_training.taxinet_dataloader import (
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
# base_waypoint_path = "/home/pulkit/trained_models_svae/taxinet_Waypoint4_Z128_svae_all_data".replace("pulkit", DEVICE_NAME)
expr_name_to_save_weights_path = {
   "orig": f"/home/pulkit/datasets_xplane/".replace("pulkit", device_name)+split+"/taxinet_wp/best_model_wp.pt"
}

vae_save_path = f"/home/pulkit/datasets_xplane/"+split+"/trained_models_svae/taxinet_Waypoint4_Z128_svae_all_data/afternoon/best_model_95.pt".replace("pulkit", DEVICE_NAME)


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


def eval_network_for_network_on_data(tod, args, model, data_dir, dataloader_params, train_dataset, split):
    # assert len(args.data_type) == 1, "Only 1 datatype at a time supported currently."
    print("Evaluating network on data")
    print(train_dataset.scaler)
    # DATALOADERS
    val_dataset, val_loader = taxinet_prepare_dataloader(
        data_dir,
        condition_str,
        "test",
        dataloader_params,
        HISTORY_LENGTH,
        train_dataset.scaler,
        [split,tod],
        args.data_type,
    )
    IMGNS = [x for x in os.listdir(data_dir) if x.endswith(".jpg") or x.endswith(".png")]
    # print(IMGNS)
    # wpt_error = np.zeros((args.num_examples, 1))
    wpt_error = np.zeros((len(IMGNS), 1))
    # print(wpt_error.shape)
    dict_wpt_err = {}
    
    for i, data in enumerate(val_loader):
        # print(data)
        if i >= args.num_examples:
            break

        scene, history, waypoint, _ = data
        scene = scene.to(device)
        history = history.to(device)
        waypoint = waypoint.to(device)
        # print("history: ", history)
        # with torch.autograd.set_detect_anomaly(True):
        sample, _ = model.vae.encode(scene)
        # print(sample.shape)
        # samples[i] = sample.clone().detach().numpy()
        # print("Sample shape: ", sample.shape)
        pred_wp, recon_scene = model.adversarial_forward(sample, history=history)
        # from IPython import embed; embed()

        wpt_error = ((waypoint - pred_wp) ** 2).mean([0,1]).detach().cpu().numpy()
        # for j in range(wpt_error.shape[0]):
            # dict_wpt_err[IMGNS[args.batch_size*i+j]] = wpt_error[j]
        dict_wpt_err[IMGNS[i]] = wpt_error
    return np.mean(np.array(list(dict_wpt_err.values())))
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
    parser = argparse.ArgumentParser(description="Args for Evaluating Networks for Robust Training")
    parser.add_argument("-expr_name", type=str, default="ok")
    parser.add_argument("-network_type", "-nt", type=str, default="orig")
    parser.add_argument("-num_examples", "-n", type=int, default=7000)
    parser.add_argument("-data_type", type=str, nargs="+", required=False)
    parser.add_argument("--batch_size", type=int, default=30)
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
    dataloader_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 24,
        "drop_last": False,
        "pin_memory": False,
    }
    WP_DATA = {}
    for tod in ["morning", "afternoon", "night"]:
        data_dir = "/home/pulkit/xplane/".replace("pulkit", DEVICE_NAME)+tod+"/"+tod +"_validation/"
        # data_dir = ("/home/pulkit/datasets_xplane/"+split+"/xplane_dataset/val/").replace("pulkit", DEVICE_NAME)
        data_dir_main = ("/home/pulkit/datasets_xplane/"+split+"/xplane_dataset/").replace("pulkit", DEVICE_NAME)
        all_images = [x for x in os.listdir(data_dir)]
        print(len(all_images))
        import xplane_training.taxinet_dataloader
        # instantiate the model and freeze all but penultimate layers
        train_dataset, train_loader = xplane_training.taxinet_dataloader.taxinet_prepare_dataloader(
            data_dir_main,   
            condition_str,
            "train",
            dataloader_params,
            HISTORY_LENGTH,
            None,
            split,
            [ExprType.standard],
        )
        # print("SIZE OF TRAINING DATA: ", len(train_dataset))
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

        WP_DATA[tod] = eval_network_for_network_on_data(tod, args, model, data_dir, dataloader_params, train_dataset, split)
   

    for tod in ["morning", "afternoon", "night"]:
        print("Model Performance on ", tod)
        print("Average Performance:", WP_DATA[tod])
        # print("Standard Deviation:", WP_DATA[tod])


        
