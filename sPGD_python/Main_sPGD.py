# ==========================================================
# Main.py
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sparse_PGD import *


import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pdb  # Debugger import
import matplotlib.animation as animation


from utilities import *


# ==========================================================
# 1) Create data for this example:
# ==========================================================

# Data should be [Dofs x Samples] and parameters should be [Parameters x Samples]


# Here add the function to create the Surface_2D.npz file:
from scipy.stats import qmc

def F_surface(x, mu):
    return np.exp(-np.abs((x - 0.5) * (mu - 1.0))) + np.sin(x * mu)

def create_surface_2D_npz(filename="Surface_2D.npz",
                          n_train=50, n_test=30,
                          x_lim=(0.0, 2.0), mu_lim=(0.0, 1.0),
                          seed_train=0, seed_test=1):
    lower = [x_lim[0], mu_lim[0]]
    upper = [x_lim[1], mu_lim[1]]

    # Latin Hypercube sampling (different seeds -> train/test don't coincide)
    s_train = qmc.LatinHypercube(d=2, seed=seed_train).random(n_train)
    s_test  = qmc.LatinHypercube(d=2, seed=seed_test).random(n_test)
    p_train = qmc.scale(s_train, lower, upper).T      # [2 x n_train]

    # Test samples scaled to the actual training interval (per parameter)
    p_min = p_train.min(axis=1)
    p_max = p_train.max(axis=1)
    p_test = qmc.scale(s_test, p_min, p_max).T        # [2 x n_test]

    # Data: [Dofs x Samples] -> 1 dof (value of F)
    x_train = F_surface(p_train[0], p_train[1])[None, :]   # [1 x n_train]
    x_test  = F_surface(p_test[0],  p_test[1])[None, :]    # [1 x n_test]

    np.savez(filename,
             x_train=x_train, x_test=x_test,
             p_train=p_train, p_test=p_test)
    print(f"Saved {filename}: x_train {x_train.shape}, p_train {p_train.shape}, "
          f"x_test {x_test.shape}, p_test {p_test.shape}")

create_surface_2D_npz("Surface_2D.npz", n_train=50, n_test=30)


# ==========================================================
# 2) Load data:
# ==========================================================

problem = "Surface_2D"
(x_train, 
    x_test, 
    p_train, 
    p_test
    ) = get_data(problem)





print("# ==================== LOADED DATA ====================")
print("Number of parameters: ", p_train.shape[0])


# ==========================================================
# 3) Compute or load sPGD:
# ==========================================================


solver = "Krig"
str_MOR = "MOR" + "_" + solver + "_" + problem + ".pkl"


if 1: # Build sPGD if it does not exist

    # ----------------------------------------------------------
    # MOR construction
    # ----------------------------------------------------------

    print("Building sPGD ...")

    pgd = SparsePGD()

    error_crit = 2.0 # Error limit.
    flag_reg = 0
    Ncp = 6 # Hyper paramter to fit.
    
    MOR = pgd.MOR_construction_sparse(
        error_crit,
        x_train,
        p_train,
        str_MOR,
        solver,
        flag_reg,
        Ncp
    )


else: # Load sPGD if it exists

    print("Loading sPGD ...")

    # ----------------------------------------------------------
    # Load MOR if it exists
    # ----------------------------------------------------------
    with open(str_MOR, "rb") as f:
        MOR = pickle.load(f)

    print("Model loaded and ready for inference.")




# ----------------------------------------------------------
# 4) Test reconstruction
# ----------------------------------------------------------

if 1: # Test reconstruction
    
    print("Performing PARAMETRIC INFERENCE on test data...")
    print("(Predicting x_test directly from p_test)")
    
    
    with torch.no_grad():

        
        x_pred = eval_MOR(MOR, p_test)
        
        i = 2 # [99 67,14,132] 132, 17, 33, 19, 60, 61, 79, 82, 83, 94, 99, 107       
        Error_tot = 100*np.linalg.norm(x_test - x_pred) / np.linalg.norm(x_test)
        print("Error_tot: ", Error_tot)
        Error = 100*np.linalg.norm(x_test[:,i] - x_pred[:,i]) / np.linalg.norm(x_test[:,i])
        print(f"Parametric Prediction Error (p→x): {Error:.4f} %")
        



    # ----------------------------------------------------------
    # Test reconstruction
    # ----------------------------------------------------------

    # --- Regular grid inside the training interval ---
    x_lim  = (p_train[0].min(), p_train[0].max())
    mu_lim = (p_train[1].min(), p_train[1].max())
    n_grid = 80
    xg, mug = np.meshgrid(np.linspace(*x_lim, n_grid), np.linspace(*mu_lim, n_grid))

    F_ref = F_surface(xg, mug)

    p_grid = np.vstack([xg.ravel(), mug.ravel()])
    with torch.no_grad():
        F_sur = np.asarray(eval_MOR(MOR, p_grid)).reshape(xg.shape)

    vmin = min(F_ref.min(), F_sur.min())
    vmax = max(F_ref.max(), F_sur.max())


    def style_axes(ax, title):
        """Style like the reference figure: mu to the right, x to the left, origin in front."""
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel("x")
        ax.set_zlabel("F")
        ax.set_title(title)
        ax.set_zlim(vmin, vmax)
        ax.set_box_aspect((1, 1, 0.7))
        ax.view_init(elev=30, azim=-135)       # origin (0,0) at the front corner
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))   # transparent panes
            axis.pane.set_edgecolor("0.6")


    fig = plt.figure(figsize=(14, 6))

    # ---- Plot 1: reference surface ----
    # Note: mu goes on the matplotlib X axis and x on the Y axis (gives the orientation of the figure)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(mug, xg, F_ref, cmap="jet", vmin=vmin, vmax=vmax,
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    style_axes(ax1, "Reference surface")

    # ---- Plot 2: surrogate surface + training points ----
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(mug, xg, F_sur, cmap="jet", vmin=vmin, vmax=vmax,
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    ax2.scatter(p_train[1], p_train[0], x_train[0], c="k", s=25,
                depthshade=False, label=f"Train points ({p_train.shape[1]})")
    style_axes(ax2, "Surrogate (sPGD) surface")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{problem}_reference_vs_surrogate.png", dpi=300, bbox_inches="tight")
    plt.show()


    """
    #pdb.set_trace()  # Debugger breakpoint

    # --- Regular grid over the domain ---
    x_lim, mu_lim = (np.min(p_train[0,:]), np.max(p_train[0,:])), (np.min(p_train[1,:]), np.max(p_train[1,:]))
    n_grid = 60
    xg, mug = np.meshgrid(np.linspace(*x_lim, n_grid), np.linspace(*mu_lim, n_grid))
    
    # Reference surface (exact function)
    F_ref = F_surface(xg, mug)

    # Surrogate surface: evaluate the MOR on the grid
    # eval_MOR expects [Parameters x Samples] -> build p_grid as [2 x n_grid^2]
    p_grid = np.vstack([xg.ravel(), mug.ravel()])
    with torch.no_grad():
        F_sur = np.asarray(eval_MOR(MOR, p_grid)).reshape(xg.shape)

    # Same z-range and color scale for both plots
    vmin = min(F_ref.min(), F_sur.min())
    vmax = max(F_ref.max(), F_sur.max())

    fig = plt.figure(figsize=(14, 6))

    # ---- Plot 1: reference surface ----
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(xg, mug, F_ref, cmap="viridis", vmin=vmin, vmax=vmax,
                    linewidth=0, antialiased=True)
    ax1.set_title("Reference surface")

    # ---- Plot 2: surrogate surface + training points ----
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(xg, mug, F_sur, cmap="viridis", vmin=vmin, vmax=vmax,
                    alpha=0.7, linewidth=0, antialiased=True)
    ax2.scatter(p_train[0], p_train[1], x_train[0], c="red", s=30,
                depthshade=False, label=f"Train points ({p_train.shape[1]})")
    ax2.set_title("Surrogate (sPGD) surface")
    ax2.legend()

    for ax in (ax1, ax2):
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\mu$")
        ax.set_zlabel("F")
        ax.set_zlim(vmin, vmax)
        ax.set_proj_type("ortho")            # orthographic projection
        ax.view_init(elev=35.264, azim=45)   # isometric view angles

    plt.tight_layout()
    plt.show()

    """





# ----------------------------------------------------------
# Inverse analysis
# ----------------------------------------------------------

if 0:
    
    print("Performing INVERSE ANALYSIS on test data...")
    
    ind = 2  # index of test data to be inverted
    x_test_i = x_test[:,ind:ind+1]  # select test data to be inverted
    p_test_ref_i = p_test[:,ind:ind+1] # reference parameter vector
    
    #pdb.set_trace()  # Debugger breakpoint

    Imposed_var = []   # no imposed parameters

    Opt_var = []
    for i in range(len(MOR["param_vec"])):
        Opt_var.append(MOR["param_vec_lim"][i])

    # MATLAB is 1-based → Python 0-based
    MOR["param_index"] = [list(range(len(MOR["param_vec"])))]

    Error, param_pred_i = Inv_LM_algo_new(MOR, x_test_i, Imposed_var, Opt_var)

    print("Error:", Error)

    
    #pdb.set_trace()  # Debugger breakpoint
    print("Reference parameters:", np.squeeze(p_test_ref_i) )
    print("Predicted parameters:", np.squeeze(param_pred_i) )


