import numpy as np
from Experiments.jrnmm import simulate_jrnmm
import torch
from scipy.signal import welch
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from Experiments.utils import train_sbi_amortized
import matplotlib.pyplot as plt

from tw_CP4SBI.baycon import BayCon
from tw_CP4SBI.scores import HPDScore
from sbi.inference import simulate_for_sbi

from tqdm import tqdm
import gc
import pickle
from copy import deepcopy
from matplotlib.patches import Patch
import os

# setting seeds for reproducibility
torch.manual_seed(75)
torch.cuda.manual_seed(75)

def model(theta):
    theta = theta.numpy()
    x = []
    for thetai in theta:
        # choose values of the JRNMM for the simulation
        C, mu, sigma = thetai
        # define timespan
        delta = 1/2**10
        burnin = 2  # given in seconds
        duration = 8  # given in seconds
        downsample = 8
        tarray = np.arange(0, burnin + duration, step=delta)
        # simulate JRNMM model with Strang splitting
        si, _ = simulate_jrnmm(mu, sigma, C, tarray, burnin, downsample)
        si = si - np.mean(si)
        _, pyyi = welch(si, nperseg=64)
        logpyyi = np.log10(pyyi)
        x.append(logpyyi)
    return torch.tensor(np.array(x))

prior = BoxUniform(
    low=torch.tensor([10.0, 50.0, 100.0]),
    high=torch.tensor([250.0, 500.0, 5000.0])
)

# defining each grid
theta_1 = torch.linspace(10.0, 250.0, 3000)
theta_2 = torch.linspace(50.0, 500.0, 3000)
theta_3 = torch.linspace(100.0, 5000.0, 3000)
theta_len = 3000

# making a grid list
theta_grid_list = [torch.cartesian_prod(theta_1, theta_2),
                   torch.cartesian_prod(theta_1, theta_3),
                   torch.cartesian_prod(theta_2, theta_3)]

# sbi checks for prior, simulator, and data consistency
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator = process_simulator(model, prior, prior_returns_numpy)
check_sbi_inputs(simulator, prior)

theta_0 = torch.tensor([135.0, 220.0, 2000.0]).view(1, -1)
x_0 = model(theta_0)

# running each nuisance parameter amortized inference
# run this chunck only once to save the density and inference list
sim_budget = 10_000
dens_list, inf_list = train_sbi_amortized(
    sim_budget=sim_budget,
    simulator=simulator,
    prior=prior,
    density_estimator='nsf',
    save_fname='Results/jrnmm_amortized',
    return_density=True,
    nuisance=True,
)

with open('Experiments/dens_list_jrnmm.pkl', 'wb') as f:
    pickle.dump(dens_list, f)

with open('Experiments/inf_list_jrnmm.pkl', 'wb') as f:
    pickle.dump(inf_list, f)

# running CP4SBI uncertainty quantification now
with open('Experiments/dens_list_jrnmm.pkl', 'rb') as f:
    dens_list = pickle.load(f)

with open('Experiments/inf_list_jrnmm.pkl', 'rb') as f:
    inf_list = pickle.load(f)


# definning calibration budgets
cal_budgets = [2000, 4000, 6000, 8000]
min_samples_leaf = [300, 300, 300, 300]
# combination list for nuisance parameters
comb_list = [[0, 1], [0, 2], [1, 2]]
device = "cpu"
uncertainty_map = {}
locart_masks = {}


# simulating samples for calibration
theta_calib, X_calib = simulate_for_sbi(
simulator, proposal=prior, num_simulations=cal_budgets[-1]
)


# for each calibration budget, simulating samples and running BayCon
for B in cal_budgets:
    print(f'Calibration budget: {B}')
    uncertainty_map[B] = []
    locart_masks[B] = []
    # after simulating, running BayCon for each nuisance parameter combination
    for i, comb in enumerate(tqdm(comb_list)):
        inference = inf_list[i]
        dens = dens_list[i]
        X_calib_used = X_calib[:B, :]
        thetas_calib_used = theta_calib[:B, comb]

        # defining the BayCon object
        baycon_obj = BayCon(
            sbi_score=HPDScore,
            base_inference=inference,
            density = dens,
            is_fitted=True,
            conformal_method="local",
            split_calib=False,
            weighting=True,
            cuda=device == "cuda",
            alpha=0.1,
        )
        
        baycon_obj.fit(
            X=None,
            theta=None,
        )
        
        # calibration step
        res = baycon_obj.locart.sbi_score.compute(X_calib_used, thetas_calib_used)

        # deriving cutoffs
        baycon_obj.calib(
            X_calib=X_calib_used,
            theta_calib=res,
            min_samples_leaf=min_samples_leaf[i],
            using_res=True,
        )

        # obtaining all cutoffs
        locart_cutoff_2d = baycon_obj.predict_cutoff(x_0)
        post_estim_2d = deepcopy(baycon_obj.locart.sbi_score.posterior)


        log_probs_obs_2d = np.exp(
            post_estim_2d.log_prob(
                x=x_0.to(device),
                theta=theta_grid_list[i].to(device),
            )
            .cpu()
            .numpy()
        )
        locart_mask_obs = -log_probs_obs_2d < locart_cutoff_2d
        locart_mask_obs = locart_mask_obs.reshape(theta_len, theta_len)

        locart_unc = baycon_obj.uncertainty_region(
            X=x_0,
            thetas=theta_grid_list[i],
            beta=0.1,
        )

        print(baycon_obj.cutoff_CI)
        print(locart_cutoff_2d)
        locart_unc = locart_unc.reshape(theta_len, theta_len)

        uncertainty_map[B].append(locart_unc)
        locart_masks[B].append(locart_mask_obs)

        # clearing the memory
        del baycon_obj
        del locart_unc
        del post_estim_2d
        del locart_cutoff_2d
        del res
        
        gc.collect()
        torch.cuda.empty_cache()

# saving results
with open('Experiments/uncertainty_map_jrnmm.pkl', 'wb') as f:
    pickle.dump(uncertainty_map, f)

with open('Experiments/locart_masks_jrnmm.pkl', 'wb') as f:
    pickle.dump(locart_masks, f)


########################### Plotting the saved results ###########################
if os.path.exists('Experiments/uncertainty_map_jrnmm.pkl') and os.path.exists('Experiments/locart_masks_jrnmm.pkl'):
    with open('Experiments/uncertainty_map_jrnmm.pkl', 'rb') as f:
        uncertainty_map = pickle.load(f)

    with open('Experiments/locart_masks_jrnmm.pkl', 'rb') as f:
        locart_masks = pickle.load(f)
else:
    raise FileNotFoundError("Required pickle files for uncertainty_map or locart_masks not found. Please run the simulation and save results first.")


def plot_uncertainty_regions_grid(
    uncertainty_map, 
    locart_masks, 
    x_lims_1,
    y_lims_1,
    x_lims = [[10.0, 250.0], [10.0, 250.0], [50.0, 500.0]],
    y_lims = [[50.0, 500.0], [100.0, 5000.0], [100.0, 5000.0]],
    theta_names=["θ₁", "θ₂", "θ₃"],
    cal_budgets=None,
    save_prefix="uncertainty_regions_jrnmm",
    ):
    """
    Plots uncertainty regions for each calibration budget and each theta pair.
    Each row corresponds to a theta pair, each column to a calibration budget.
    """

    if cal_budgets is None:
        cal_budgets = sorted(list(uncertainty_map.keys()))
    comb_labels = [
        f"{theta_names[0]} vs {theta_names[1]}",
        f"{theta_names[0]} vs {theta_names[2]}",
        f"{theta_names[1]} vs {theta_names[2]}"
    ]

    plt.style.use('dark_background')
    fig, axes = plt.subplots(
        3, len(cal_budgets), figsize=(5 * len(cal_budgets), 15),
        squeeze=False
    )
    fig.patch.set_facecolor('black')

    for row_idx in range(3):
        for col_idx, B in enumerate(cal_budgets):
            ax = axes[row_idx, col_idx]
            locart_unc = uncertainty_map[B][row_idx]
            locart_mask_obs = locart_masks[B][row_idx]

            # Plot mask contour
            ax.contour(
                locart_mask_obs.T,
                extent=(x_lims[row_idx][0], x_lims[row_idx][1], y_lims[row_idx][0], y_lims[row_idx][1]),
                levels=[0.5],
                colors="dodgerblue",
                linewidths=2,
                alpha=1.0,
            )
            # Plot inside region
            ax.contourf(
                locart_unc.T,
                extent=(x_lims[row_idx][0], x_lims[row_idx][1], y_lims[row_idx][0], y_lims[row_idx][1]),
                levels=[0.99, 1.01],
                colors="lime",
                linewidths=2,
                alpha=0.25,
            )
            # Plot underterminate region
            ax.contourf(
                locart_unc.T,
                extent=(x_lims[row_idx][0], x_lims[row_idx][1], y_lims[row_idx][0], y_lims[row_idx][1]),
                levels=[0.49, 0.51],
                colors="darkorange",
                alpha=0.8,
            )

            ax.set_title(f"{comb_labels[row_idx]}, B={B}", fontsize=14)
            ax.set_xlabel(comb_labels[row_idx].split(" vs ")[0], fontsize=12)
            ax.set_ylabel(comb_labels[row_idx].split(" vs ")[1], fontsize=12)
            ax.set_xlim(x_lims_1[row_idx][0], x_lims_1[row_idx][1])
            ax.set_ylim(y_lims_1[row_idx][0], y_lims_1[row_idx][1])

            del locart_unc, locart_mask_obs, ax
            gc.collect()
            torch.cuda.empty_cache()

    legend_elements = [
        Patch(facecolor="none", edgecolor="dodgerblue", linewidth=2, label="CP4SBI-LOCART", alpha=0.75),
        Patch(facecolor="lime", edgecolor="none", linewidth=2, label="Inside region", alpha=0.25),
        Patch(facecolor="darkorange", edgecolor="none", linewidth=2, label="Underterminate region", alpha=0.8),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(legend_elements),
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.925)
    plt.rcParams.update({"font.size": 16})
    fig.savefig(f"{save_prefix}_grid.pdf", dpi=300)

    plt.close(fig)
    gc.collect()
    del fig, axes, legend_elements
    gc.collect()
    torch.cuda.empty_cache()

# Count the number of entries equal to 1 in one uncertainty map (for example, for the largest calibration budget and first theta pair)
B = cal_budgets[-1]  # Use the largest calibration budget
row_idx = 2          # Use the second theta pair (θ₂ vs θ₃)
uncertainty_region = uncertainty_map[B][row_idx]
num_entries_equal_1 = np.sum(uncertainty_region == 1)
print(f"Number of entries equal to 1 in uncertainty_map[{B}][{row_idx}]: {num_entries_equal_1}")

x_lims_1 = [[125, 142.5], [125, 145.0], [160.0, 250.0]]
y_lims_1 = [[170, 260.0], [1750, 2350.0], [1750, 2350.0]]

plot_uncertainty_regions_grid(
    uncertainty_map,
    locart_masks,
    cal_budgets=[2000, 4000, 6000],
    x_lims_1=x_lims_1,
    y_lims_1=y_lims_1,
)

################# Plot to be used in the paper #################
# Plotting only for the largest calibration budget, all projections in the same row
B = cal_budgets[-1]  # Largest calibration budget
comb_labels = [
    r"$\theta_1$ vs $\theta_2$",
    r"$\theta_1$ vs $\theta_3$",
    r"$\theta_2$ vs $\theta_3$"
]
x_lims_1 = [[115, 155], [115, 155.0], [135.0, 255.0]]
y_lims_1 = [[135, 255.0], [1700, 2400.0], [1700, 2400.0]]

x_lims = [[10.0, 250.0], [10.0, 250.0], [50.0, 500.0]]
y_lims = [[50.0, 500.0], [100.0, 5000.0], [100.0, 5000.0]]

plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
fig.patch.set_facecolor('black')
plt.rcParams.update({"font.size": 16})
plt.rcParams.update({"legend.fontsize": 14})

for idx in range(3):
    ax = axes[0, idx]
    locart_unc = uncertainty_map[B][idx]
    locart_mask_obs = locart_masks[B][idx]

    ax.contour(
        locart_mask_obs.T,
        extent=(x_lims[idx][0], x_lims[idx][1], y_lims[idx][0], y_lims[idx][1]),
        levels=[0.5],
        colors="dodgerblue",
        linewidths=2,
        alpha=1.0,
    )
    ax.contourf(
        locart_unc.T,
        extent=(x_lims[idx][0], x_lims[idx][1], y_lims[idx][0], y_lims[idx][1]),
        levels=[0.99, 1.01],
        colors="lime",
        linewidths=2,
        alpha=0.25,
    )
    ax.contourf(
        locart_unc.T,
        extent=(x_lims[idx][0], x_lims[idx][1], y_lims[idx][0], y_lims[idx][1]),
        levels=[0.49, 0.51],
        colors="darkorange",
        alpha=0.8,
    )

    ax.set_title(f"{comb_labels[idx]}, B={B}", fontsize=16)
    xlabel = comb_labels[idx].split(" vs ")[0]
    ylabel = comb_labels[idx].split(" vs ")[1]
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlim(x_lims_1[idx][0], x_lims_1[idx][1])
    ax.set_ylim(y_lims_1[idx][0], y_lims_1[idx][1])

    del locart_unc, locart_mask_obs, ax
    gc.collect()
    torch.cuda.empty_cache()

legend_elements = [
    Patch(facecolor="none", edgecolor="dodgerblue", linewidth=2, label="CP4SBI", alpha=0.75),
    Patch(facecolor="lime", edgecolor="none", linewidth=2, label="Inside region", alpha=0.25),
    Patch(facecolor="darkorange", edgecolor="none", linewidth=2, label="Undetermined region", alpha=0.8),
]
fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.005),
    ncol=len(legend_elements),
    frameon=False,
)

plt.tight_layout()
plt.subplots_adjust(top=0.825)
plt.rcParams.update({"font.size": 16})
fig.savefig(f"uncertainty_regions_jrnmm_largest_budget_row.pdf", dpi=300)

plt.close(fig)
gc.collect()
del fig, axes, legend_elements
gc.collect()
torch.cuda.empty_cache()


