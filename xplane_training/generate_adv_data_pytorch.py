#
#  python3 xplane/generate_adv_data.py -expr_name grad_test -num_examples 500 -data_type orig -waypoint_path xplane/scratch/taxinet_Waypoint4_Z128_svae_all_data/afternoon/best_model.pt --save_plot
#

import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from IPython import embed

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from forecaster import AdverLinear, AdverNonLinear

# from planners.xplane_mpc_double_integrator import track_traj, NX, TARGET_SPEED, T
import torch.nn.functional as F
from xplane.adv_eval_utils import (
    get_adv_args,
    init_data_loaders,
    init_models,
    load_wpt_weights,
    GenerateDataNps,
    model_forward_and_mpc,
    save_images,
)
from xplane.utils import ENDC, OKBLUE, OKRED, OKYELLOW, SEED, remove_and_create_dir

# make sure this is a system variable in your bashrc
# NASA_ULI_ROOT_DIR = os.environ["NASA_ULI_ROOT_DIR"]

# DATA_DIR = os.environ["NASA_DATA_DIR"]

# where intermediate results are saved
# never save this to the main git repo
SCRATCH_DIR = "./adver_data/"

torch.manual_seed(SEED)

EXAMPLE_NUM = -1


Z_DIM = 128
H = Z_DIM
P = Z_DIM
# LAMBDA = 1e-5
LAMBDA = 0.0
KAPPA = 2.0
LR = 0.01


torch.cuda.empty_cache()
device = torch.device("cpu")

base_path = "./xplane/"
args = get_adv_args()

print(OKYELLOW + f"Running {args.expr_name} experiment" + ENDC)
print(OKYELLOW + f"with params KAPPA: {KAPPA}, Learning Rate: {LR}" + ENDC)


condition_str = "afternoon"
results_dir = remove_and_create_dir(
    SCRATCH_DIR
    + f"/expr_{args.expr_name}_{args.num_examples}_LR_{LR}_K{KAPPA}_Z_{Z_DIM}_{condition_str}/"
)
if EXAMPLE_NUM != -1:
    results_dir = remove_and_create_dir(
        SCRATCH_DIR + f"/adver_results/expr_{args.expr_name}_{EXAMPLE_NUM}_LR_{LR}_K{KAPPA}_Z_{Z_DIM}_{condition_str}/"
    )

# vae_save_path = os.path.join(
#     base_path,
#     f"scratch/taxinet_VAE_Z{Z_DIM}_train/{condition_str}/best_model_45.pt",
# )
# vae_save_path = f"xplane/scratch/taxinet_VAE_Z{Z_DIM}_only_afternoon/afternoon/best_model.pt"
vae_save_path = f"./trained_models/taxinet_VAE_Z{Z_DIM}_all_data/{condition_str}/best_model_55.pt"

PATH_LENGTH = 1
HISTORY_LENGTH = 4

_, wpt_model = init_models(Z_DIM, device, HISTORY_LENGTH, PATH_LENGTH, vae_save_path)
wpt_model = load_wpt_weights(wpt_model, args, device)

orig_train_dataset, train_loader, val_loader = init_data_loaders(args, HISTORY_LENGTH, condition_str)

expr_data = GenerateDataNps(args, Z_DIM, HISTORY_LENGTH, PATH_LENGTH, EXAMPLE_NUM)

print(OKRED + f"Using SEED: {SEED}" + ENDC)
print("Adversarial Training")


for idx, data in enumerate(train_loader):

    if idx >= args.num_examples:
        break

    if EXAMPLE_NUM != -1 and idx != EXAMPLE_NUM:
        continue

    forecaster = AdverLinear(size=H)
    optimizer = torch.optim.SGD(forecaster.parameters(), lr=LR)

    scene, history, waypoint, _ = data
    scene = scene.to(device)
    history = history.to(device)
    waypoint = waypoint.to(device)

    # scene.requires_grad = False
    # history.requires_grad = False
    # waypoint.requires_grad = False

    expr_data.adv_history[idx] = history.clone().reshape(-1, HISTORY_LENGTH, 2)[0]
    expr_data.adv_waypoint[idx] = waypoint.clone().reshape(-1, PATH_LENGTH, 2)[0]

    sample, logvar = wpt_model.vae.encode(scene)
    prev_sample = sample.clone()

    # distrib = torch.distributions.multivariate_normal.MultivariateNormal(
    #     prev_sample, torch.eye(Z_DIM) * torch.exp(logvar.squeeze())
    # )
    # orig_prob = torch.exp(distrib.log_prob(prev_sample))

    best_adv_cost = -np.inf
    best_cost_pred_waypoints = None
    orig_mpc_cost = None

    for jdx in range(args.num_grad_steps):

        mpc_costs, opt_state, opt_u, mpc_pred_wp, recon_scene = model_forward_and_mpc(
            wpt_model, sample, history, PATH_LENGTH, orig_train_dataset, waypoint.clone()
        )
        if jdx == 0:
            orig_mpc_cost = mpc_costs.total.item()
            # expr_data.adv_controls[idx, 0, :, :] = opt_u.detach().T

        mse_loss = torch.nn.MSELoss()
        consistency_loss = mse_loss(sample, prev_sample)

        # adv_prob = torch.max(torch.exp(distrib.log_prob(sample)), orig_prob * 1e-5)
        # print(adv_prob, sample.mean(), orig_prob, prev_sample.mean())
        # consistency_loss = adv_prob * (adv_prob / orig_prob).log()

        reg_loss = torch.autograd.Variable(torch.FloatTensor(1), requires_grad=True)
        for W in forecaster.parameters():
            reg_loss = reg_loss + W.norm(2)

        loss = mpc_costs.total - KAPPA * consistency_loss - LAMBDA * reg_loss

        neg_loss = -loss
        neg_loss.backward()
        optimizer.step()

        if mpc_costs.total > best_adv_cost:
            expr_data.set(idx, recon_scene, mpc_costs, sample, opt_u)
            best_cost_pred_waypoints = mpc_pred_wp
            best_adv_cost = mpc_costs.total

            print(
                OKYELLOW
                + f"Example {idx}, Grad Step: {jdx}, Cost: {mpc_costs.total.item():.2f}"
                + ENDC
                + f", Tracking: {mpc_costs.tracking:.2f}, Control: {mpc_costs.control:.2f},"
                f" Goal: {mpc_costs.goal:.2f}, Loss: {loss.item():.2f},"
                f" Consist: {consistency_loss.item():.2f}, Reg Loss: {reg_loss.item():.2f}"
            )
        else:
            print(
                f"Example {idx}, Grad Step: {jdx}, Cost: {mpc_costs.total.item():.2f}, Loss: {loss.item():.2f}"
                + f" Consist: {consistency_loss.item():.2f}, Reg Loss: {reg_loss.item():.2f}"
            )

        if EXAMPLE_NUM != -1:
            expr_data.adv_loss[0, jdx, 0] = mpc_costs.total
            expr_data.adv_loss[0, jdx, 1] = KAPPA * consistency_loss
            expr_data.adv_loss[0, jdx, 2] = reg_loss
        else:
            expr_data.adv_loss[idx, jdx, 0] = mpc_costs.total
            expr_data.adv_loss[idx, jdx, 1] = KAPPA * consistency_loss
            expr_data.adv_loss[idx, jdx, 2] = reg_loss

        # prev_sample = sample.clone().detach()
        # Forecaster forward
        sample = forecaster(prev_sample)

    if args.save_plot:
        save_images(
            (scene, orig_train_dataset.inverse_transform_torch(waypoint.clone()), orig_mpc_cost),
            (expr_data.adv_scenes[idx], best_cost_pred_waypoints, best_adv_cost.item()),
            expr_data.adv_loss[0] if EXAMPLE_NUM != -1 else expr_data.adv_loss[idx],
            # save_path=results_dir + f"/adv_recon_{SEED}_{str(idx)}_{KAPPA}_{LR}.png",
            save_path=results_dir,
            idx=idx,
        )

    print("\n")

expr_data.print()
expr_data.save(args, SEED, EXAMPLE_NUM, KAPPA, LR)
