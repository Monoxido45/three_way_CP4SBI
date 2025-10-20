# posterior_npe_two_moons_sbibm.py
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 16})

from sbi.utils import BoxUniform
from sbi.inference import SNPE
from tw_CP4SBI.baycon import BayCon
from tw_CP4SBI.scores import HPDScore
import gc

from copy import deepcopy
# removed unused imports: BayCon, HPDScore

# sbibm import
from sbibm.tasks import get_task
import torch


# sbi imports + fixed grid utilities
# Replace the previous grid_from_samples / logprob_grid with a fixed grid in [-1, 1]
def grid_fixed(nbins=200, low=-1.0, high=1.0):
    xi = np.linspace(low, high, nbins)
    yi = np.linspace(low, high, nbins)
    X, Y = np.meshgrid(xi, yi)
    return xi, yi, X, Y


def logprob_grid(posterior, x_o, nbins=200, low=-1.0, high=1.0, device="cpu"):
    xi, yi, X, Y = grid_fixed(nbins=nbins, low=low, high=high)
    grid_coords = np.vstack([X.ravel(), Y.ravel()]).T  # (N, 2)
    grid_torch = torch.from_numpy(grid_coords.astype(np.float32)).to(device)
    x_o_t = x_o.to(device)
    with torch.no_grad():
        logp = posterior.log_prob(grid_torch, x=x_o_t)  # (N,)
    logp_np = logp.detach().cpu().numpy()
    logp_np = logp_np - np.max(logp_np)
    Z = np.exp(logp_np).reshape(X.shape)
    return xi, yi, X, Y, Z

def plot_posterior(X, Y, Z, out="posterior_npe_two_moons_sbibm.png", cmap="viridis", xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(4.5, 4))

    # extent from meshgrid: use xi/yi extremes for accurate pixel alignment
    extent = [X[0, 0], X[0, -1], Y[0, 0], Y[-1, 0]]

    # Use nearest interpolation to avoid contour-like artifacts; do not assign the returned AxesImage
    ax.imshow(
        Z,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {out}")
    plt.show()

def main():
    device = "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    # Load sbibm task (two_moons)
    task = get_task("two_moons")
    sbibm_simulator = task.get_simulator()
    sbibm_prior = task.get_prior()

    # setting sbi prior
    prior = BoxUniform(
            low=-1 * torch.ones(2),
            high=1 * torch.ones(2),
            device=device,
        )

    # Simulate training data using sbibm simulator and sbibm prior
    num_simulations = 2000
    theta =  sbibm_prior(num_samples= num_simulations)
    x = sbibm_simulator(theta)

    # Fit NPE (SNPE)
    inference = SNPE(prior=prior, density_estimator="mdn")
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)

    # Get observed data from sbibm task
    x_o = task.get_observation(num_observation=2)

    # Optional: show posterior heatmap once (kept as a quick diagnostic)
    xi, yi, X, Y, Z = logprob_grid(posterior, x_o, nbins=1000, low=-1.0, high=1.0, device=device)
    plot_posterior(X, Y, Z, out="posterior_npe_two_moons_sbibm.png")
    # Also save a zoomed-in density image on the upper blob
    try:
        upper_mask0 = (Y > 0)
        masked_Z0 = np.where(upper_mask0, Z, -np.inf)
        iy0, ix0 = np.unravel_index(np.argmax(masked_Z0), masked_Z0.shape)
        cx0, cy0 = float(X[iy0, ix0]), float(Y[iy0, ix0])
    except Exception:
        iy0, ix0 = np.unravel_index(np.argmax(Z), Z.shape)
        cx0, cy0 = float(X[iy0, ix0]), float(Y[iy0, ix0])

    
    wx0, wy0 = 0.20, 0.20
    xlim0 = (max(-1.0, cx0 - wx0), min(1.0, cx0 + wx0))
    ylim0 = (max(-1.0, cy0 - wy0), min(1.0, cy0 + wy0))
    plot_posterior(X, Y, Z, out="posterior_npe_two_moons_sbibm_zoom.png", xlim=xlim0, ylim=ylim0)

    # based on it, obtaining an calibrated HPD with few samples
    B = 1000
    min_samples_leaf = 150
    theta_calib = sbibm_prior(num_samples=B)
    x_calib = sbibm_simulator(theta_calib)

    # defining the BayCon object
    baycon_obj = BayCon(
        sbi_score=HPDScore,
        base_inference=inference,
        density=density_estimator,
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
    res = baycon_obj.locart.sbi_score.compute(x_calib, theta_calib)

    # deriving cutoffs
    baycon_obj.calib(
        X_calib=x_calib,
        theta_calib=res,
        min_samples_leaf=min_samples_leaf,
        using_res=True,
    )

    # obtaining cutoff
    locart_cutoff_2d = baycon_obj.predict_cutoff(x_o)
    post_estim_2d = deepcopy(baycon_obj.locart.sbi_score.posterior)

    # Compute and plot CP4SBI-style credible region
    # 1) Build a higher-resolution grid in [-1, 1]
    grid_nbins = 2000  # increase mesh for smoother contours/regions
    _, _, Xg, Yg = grid_fixed(nbins=grid_nbins, low=-1.0, high=1.0)
    grid_coords = np.vstack([Xg.ravel(), Yg.ravel()]).T.astype(np.float32)

    # 2) Evaluate posterior density on the grid (batched for performance/memory)
    batch_size = 200_000
    logp_list = []
    x_o_dev = x_o.to(device)
    with torch.no_grad():
        for start in range(0, grid_coords.shape[0], batch_size):
            end = min(start + batch_size, grid_coords.shape[0])
            grid_torch = torch.from_numpy(grid_coords[start:end]).to(device)
            logp_chunk = post_estim_2d.log_prob(grid_torch, x=x_o_dev)
            logp_list.append(logp_chunk.detach().cpu())
    logp_grid = torch.cat(logp_list, dim=0)
    dens_grid = np.exp(logp_grid.numpy()).reshape(Xg.shape)

    # 3) Build CP4SBI mask on the grid using the learned cutoff
    try:
        locart_cutoff_val = float(locart_cutoff_2d)
    except Exception:
        locart_cutoff_val = locart_cutoff_2d.item() if hasattr(locart_cutoff_2d, "item") else float(locart_cutoff_2d)

    locart_mask_obs = (-dens_grid) < locart_cutoff_val

    # 4) Uncertainty map from BayCon (inside=1, undetermined=0) evaluated in batches
    alpha = 0.1
    unc_list = []
    for start in range(0, grid_coords.shape[0], batch_size):
        end = min(start + batch_size, grid_coords.shape[0])
        thetas_chunk = torch.from_numpy(grid_coords[start:end]).to(device)
        unc_chunk = baycon_obj.uncertainty_region(
            X=x_o,
            thetas=thetas_chunk,
            beta=alpha,
            strategy="assymetric",
        )
        # Handle both torch.Tensor and numpy.ndarray returns
        if isinstance(unc_chunk, torch.Tensor):
            arr = unc_chunk.detach().cpu().numpy()
        else:
            arr = np.asarray(unc_chunk)
        # Ensure 1D shape (batch_len,)
        arr = np.ravel(arr)
        # Optionally validate expected length
        # assert arr.shape[0] == (end - start), f"Unexpected chunk size: {arr.shape} vs {(end-start)}"
        unc_list.append(arr)
    locart_unc = np.concatenate(unc_list, axis=0).reshape(Xg.shape)
    
    # Plot cutoff CI as an errorbar with the predicted cutoff point
    ci = getattr(baycon_obj, "cutoff_CI", None)
    cutoff_val = locart_cutoff_2d

    if ci is not None:
        lo, hi = ci[:, 0], ci[:, 1]
        lo = float(lo)
        hi = float(hi)
        lower_err = cutoff_val - lo
        upper_err = hi - cutoff_val

        fig_cut, ax_cut = plt.subplots(figsize=(3.2, 2.8))
        ax_cut.errorbar(
            [0],
            [cutoff_val],
            yerr=[[lower_err], [upper_err]],
            fmt="o",
            color="#2979ff",
            ecolor="#2979ff",
            elinewidth=2,
            capsize=5,
        )
        ax_cut.set_xlim(-1, 1)
        ax_cut.set_xticks([])
        ax_cut.set_ylabel(r"$t(\mathbf{x})$")
        ax_cut.grid(alpha=0.2)
        plt.tight_layout()
        out_cut = "posterior_cutoff_CI_errorplot.png"
        fig_cut.savefig(out_cut, dpi=400, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved cutoff CI errorplot to {out_cut}")
        plt.close(fig_cut)



    # Determine zoom window around the upper blob (maximize density for y > 0)
    try:
        upper_mask = (Yg > 0)
        masked_dens = np.where(upper_mask, dens_grid, -np.inf)
        iy, ix = np.unravel_index(np.argmax(masked_dens), masked_dens.shape)
        cx, cy = float(Xg[iy, ix]), float(Yg[iy, ix])
    except Exception:
        iy, ix = np.unravel_index(np.argmax(dens_grid), dens_grid.shape)
        cx, cy = float(Xg[iy, ix]), float(Yg[iy, ix])

    # Zoom extents (tune half-widths as needed)
    wx, wy = 0.20, 0.20
    x_min, x_max = max(-1.0, cx - wx), min(1.0, cx + wx)
    y_min, y_max = max(-1.0, cy - wy), min(1.0, cy + wy)

    # Prepare cutoff and optional CI levels in density units (score = -density)
    cutoff_level = -float(locart_cutoff_val)
    cutoff_CI = getattr(baycon_obj, "cutoff_CI", None)
    dens_CI_levels = None
    if cutoff_CI is not None:
        try:
            lo, hi = cutoff_CI
            dens_CI_levels = [ -float(lo), -float(hi) ]
        except Exception:
            dens_CI_levels = None

    # 5) Produce two separate figures per request (no dark background, no legends, no target region)
    # (a) Only CP4SBI contour line (blue) with viridis density background
    fig1, ax1 = plt.subplots(figsize=(4.0, 4.0))
    # adding to remove colors from outside the threshold
    threshold = 0.05
    nb = 256
    base = plt.get_cmap("viridis", nb)
    colors = base(np.linspace(0, 1, nb))
    cut = int(np.round(threshold * (nb - 1)))
    colors[:cut, :] = np.array([1.0, 1.0, 1.0, 1.0])  # white for values < threshold
    cmap_custom = mcolors.ListedColormap(colors)

    # Add viridis density background
    ax1.imshow(
        dens_grid,
        origin='lower',
        extent=(-1, 1, -1, 1),
        cmap=cmap_custom,
        alpha=1.0,
        interpolation='bilinear',
        zorder=0,
    )
    # Highlight CP4SBI boundary using density level set at the cutoff
    ax1.contour(
        dens_grid,
        levels=[cutoff_level],
        extent=(-1, 1, -1, 1),
        colors="#2979ff",
        linewidths=1.4,
        alpha=1.0,
        zorder=3,
    )
    # Add CI boundaries if available
    if dens_CI_levels is not None:
        for lvl in dens_CI_levels:
            try:
                ax1.contour(
                    dens_grid,
                    levels=[lvl],
                    extent=(-1, 1, -1, 1),
                    colors="#2979ff",
                    linewidths=1.0,
                    linestyles="--",
                    alpha=0.8,
                    zorder=3,
                )
            except Exception:
                pass
    ax1.set_xlabel(r"$\theta_1$")
    ax1.set_ylabel(r"$\theta_2$")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    plt.tight_layout()
    out_contour = "posterior_cp4sbi_contour_only.png"
    fig1.savefig(out_contour, dpi=400, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved CP4SBI contour-only plot to {out_contour}")

    # (b) CP4SBI contour + inside (lime) and undetermined (darkorange)
    fig2, ax2 = plt.subplots(figsize=(4.0, 4.0))
    ax2.contourf(
        locart_unc,
        levels=[0.99, 1.01],
        extent=(-1, 1, -1, 1),
        colors="forestgreen",
        alpha=0.55,
        zorder=1,
    )
    ax2.contourf(
        locart_unc,
        levels=[0.49, 0.51],
        extent=(-1, 1, -1, 1),
        colors="darkorange",
        alpha=0.8,
        zorder=2,
    )
    # Overlay CP4SBI boundary and CI on top
    ax2.contour(
        dens_grid,
        levels=[cutoff_level],
        extent=(-1, 1, -1, 1),
        colors="#2979ff",
        linewidths=1.2,
        alpha=1.0,
        zorder=4,
    )
    if dens_CI_levels is not None:
        for lvl in dens_CI_levels:
            try:
                ax2.contour(
                    dens_grid,
                    levels=[lvl],
                    extent=(-1, 1, -1, 1),
                    colors="#2979ff",
                    linewidths=0.9,
                    linestyles="--",
                    alpha=0.9,
                    zorder=4,
                )
            except Exception:
                pass
    ax2.set_xlabel(r"$\theta_1$")
    ax2.set_ylabel(r"$\theta_2$")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    plt.tight_layout()
    out_regions = "posterior_cp4sbi_contour_with_regions.png"
    fig2.savefig(out_regions, dpi=400, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved CP4SBI contour-with-regions plot to {out_regions}")

    # (c) Optional: CI-only figure (cutoff + CI bounds), zoomed to upper blob
    try:
        # Determine zoom window via dens_grid maximum for y > 0
        upper_mask = (Yg > 0)
        masked_dens = np.where(upper_mask, dens_grid, -np.inf)
        iy, ix = np.unravel_index(np.argmax(masked_dens), masked_dens.shape)
        cx, cy = float(Xg[iy, ix]), float(Yg[iy, ix])
    except Exception:
        iy, ix = np.unravel_index(np.argmax(dens_grid), dens_grid.shape)
        cx, cy = float(Xg[iy, ix]), float(Yg[iy, ix])

    wx, wy = 0.20, 0.20
    x_min_ci, x_max_ci = max(-1.0, cx - wx), min(1.0, cx + wx)
    y_min_ci, y_max_ci = max(-1.0, cy - wy), min(1.0, cy + wy)

    # Convert cutoff and CI to density levels (score = -density)
    cutoff_val = float(locart_cutoff_2d)
    cutoff_level = -cutoff_val
    ci = getattr(baycon_obj, "cutoff_CI", None)
    dens_CI_levels = None
    if ci is not None:
        try:
            lo, hi = ci
            dens_CI_levels = [-float(lo), -float(hi)]
        except Exception:
            dens_CI_levels = None

    fig3, ax3 = plt.subplots(figsize=(4.0, 4.0))
    # Plot cutoff (solid)
    ax3.contour(
        dens_grid,
        levels=[cutoff_level],
        extent=(-1, 1, -1, 1),
        colors="#2979ff",
        linewidths=1.4,
        alpha=1.0,
    )
    # Plot CI bounds (dashed)
    if dens_CI_levels is not None:
        for lvl in dens_CI_levels:
            try:
                ax3.contour(
                    dens_grid,
                    levels=[lvl],
                    extent=(-1, 1, -1, 1),
                    colors="#2979ff",
                    linewidths=1.0,
                    linestyles="--",
                    alpha=0.9,
                )
            except Exception:
                pass
    ax3.set_xlabel(r"$\theta_1$")
    ax3.set_ylabel(r"$\theta_2$")
    ax3.set_xlim(x_min_ci, x_max_ci)
    ax3.set_ylim(y_min_ci, y_max_ci)
    plt.tight_layout()
    out_ci = "posterior_cp4sbi_ci_only.png"
    fig3.savefig(out_ci, dpi=400, bbox_inches='tight', pad_inches=0.02)

    # (d) Separate CI as an errorbar plot
    ci = getattr(baycon_obj, "cutoff_CI", None)
    if ci is not None:
        try:
            lo, hi = ci
            cutoff_val_scalar = float(locart_cutoff_2d)
            yerr_lower = cutoff_val_scalar - float(lo)
            yerr_upper = float(hi) - cutoff_val_scalar
            fig4, ax4 = plt.subplots(figsize=(3.2, 2.8))
            ax4.errorbar([0], [cutoff_val_scalar], yerr=[[yerr_lower], [yerr_upper]], fmt='o', color='#2979ff', ecolor='#2979ff', elinewidth=2, capsize=5)
            ax4.set_xlim(-1, 1)
            ax4.set_xticks([])
            ax4.set_ylabel("score cutoff (−density)")
            ax4.grid(alpha=0.2)
            plt.tight_layout()
            out_err = "posterior_cp4sbi_cutoff_errorbar.png"
            fig4.savefig(out_err, dpi=400, bbox_inches='tight', pad_inches=0.02)
            print(f"Saved CP4SBI cutoff errorbar plot to {out_err}")
            plt.close(fig4)
        except Exception:
            pass
    print(f"Saved CP4SBI CI-only plot to {out_ci}")

    # cleanup
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    del fig1, fig2, fig3, ax1, ax2, ax3, locart_unc, locart_mask_obs, dens_grid, logp_grid
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
