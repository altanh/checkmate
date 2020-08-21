import argparse
import logging
import os
import pathlib
import pickle
import shutil
import uuid
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import seaborn as sns
import tensorflow as tf
from matplotlib.lines import Line2D
from scipy.stats.mstats import gmean

from experiments.common.definitions import remat_data_dir
from experiments.common.graph_plotting import render_dfgraph
from experiments.common.load_keras_model import MODEL_NAMES, get_keras_model, CHAIN_GRAPH_MODELS
from experiments.common.profile.cost_model import CostModel
from experiments.common.profile.platforms import PLATFORM_CHOICES, platform_memory, pretty_platform_name
from experiments.common.ray_utils import get_futures
from remat.core.dfgraph import DFGraph
from remat.core.enum_strategy import SolveStrategy
from remat.core.schedule import ScheduledResult
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all, solve_checkpoint_all_ap
from remat.core.solvers.strategy_checkpoint_last import solve_checkpoint_last_node
from remat.core.solvers.strategy_chen import solve_chen_sqrtn, solve_chen_greedy
from remat.core.solvers.strategy_griewank import solve_griewank, clean_griewank_cache
from remat.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi
from remat.core.solvers.strategy_simrd import solve_simrd
from remat.tensorflow2.extraction import dfgraph_from_keras

from simrd.heuristic import DTREqClass
from simrd.runtime import RuntimeV2EagerOptimized

NUM_ILP_CORES = os.environ.get("ILP_CORES", 12 if os.cpu_count() > 12 else 4)

# Budget selection parameters
NUM_ILP_GLOBAL = 32
NUM_ILP_LOCAL = 32
ILP_SEARCH_RANGE = [0.5, 1.5]
ILP_ROUND_FACTOR = 1000  # 1KB
PLOT_UNIT_RAM = 1e9  # 9 = GB
DENSE_SOLVE_MODELS = ["VGG16", "VGG19"]
ADDITIONAL_ILP_LOCAL_POINTS = {  # additional budgets to add to ILP local search.
    # measured in GB, including fixed parameters.
    ("ResNet50", 256): [9.25, 9.5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ("MobileNet", 512): [15.9, 15, 14, 17, 18, 19, 38, 37, 36, 39],
    ("vgg_unet", 32): [20, 21, 22, 23, 24, 25, 26, 6, 7, 8, 9, 10, 11, 12, 16, 15.9]
}

# Plotting parameters
XLIM = {
    "MobileNet": [4, 48],
    "vgg_unet": [6, 39],
    "VGG16": [13, 22]
}

YLIM = {
    "MobileNet": [0.95, 1.5],
    "vgg_unet": [0.95, 1.45],
    "VGG16": [0.95, 1.5]
}

KEYS = {
  "VGG16": "p32xlarge_VGG16_256_None",
  "MobileNet": "p32xlarge_MobileNet_512_None",
  "vgg_unet": "p32xlarge_vgg_unet_32_None"
}

TITLES = {
  "VGG16": "VGG16 (256)",
  "MobileNet": "MobileNet (512)",
  "vgg_unet": "U-Net (32)"
}

INPUT_SHAPES = {
  "VGG16": "224x224",
  "MobileNet": "224x224",
  "vgg_unet": "416x608"
}

INCLUDE_STRATEGIES = {
  SolveStrategy.CHEN_SQRTN, SolveStrategy.CHEN_SQRTN_NOAP,
  SolveStrategy.CHEN_GREEDY, SolveStrategy.CHEN_GREEDY_NOAP,
  SolveStrategy.GRIEWANK_LOGN,
  SolveStrategy.OPTIMAL_ILP_GC,
  SolveStrategy.SIMRD
}

CHECKPOINT_ALL_MEMORY = {
  "VGG16": 20.692138304,
  "MobileNet": 37.859827008,
  "vgg_unet": 23.048189592
}

LEGEND_LABELS = {
  SolveStrategy.CHEN_GREEDY: "Chen et al. greedy",
  SolveStrategy.CHEN_SQRTN: r"Chen et al. $\sqrt{n}$",
  SolveStrategy.GRIEWANK_LOGN: r"Griewank & Walther $\log(n)$",
  SolveStrategy.CHECKPOINT_ALL: "Checkpoint all (ideal)",
  SolveStrategy.OPTIMAL_ILP_GC: "Checkmate (optimal ILP)",
  SolveStrategy.SIMRD: "DTR (EqClass)"
}

def plot_model(model: str, fig, ax):
  log_base = remat_data_dir() / "budget_sweep" / KEYS[model]
  results = pickle.load((log_base / "export_prefix_min_data.pickle").open("rb"))

  for solve_strategy, points in results.items():
    solve_strategy = eval("SolveStrategy.{}".format(solve_strategy))
    if solve_strategy not in INCLUDE_STRATEGIES: continue

    xs, ys = zip(*points)
    xs, ys = list(xs), list(ys)
    xs[-1] = 100 # put it off the plot like in Checkmate's graphic
    color, marker, markersize = SolveStrategy.get_plot_params(solve_strategy)
    ax.step(xs, ys, where="post", zorder=1, color=color)
    scatter_zorder = 2
    ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], label="", zorder=scatter_zorder, s=markersize**2, color=color, marker=marker)
    ax.set_xlim(XLIM[model])
    ax.set_ylim(YLIM[model])

  # annotate different adapations if U-Net
  if model == "vgg_unet":
    color, _, _ = SolveStrategy.get_plot_params(SolveStrategy.CHEN_SQRTN_NOAP)
    ax.annotate("*", xy=results["CHEN_SQRTN_NOAP"][0], xytext=(-10, -6), textcoords="offset points", color=color)
    ax.annotate("**", xy=results["CHEN_SQRTN"][0], xytext=(-15, -6), textcoords="offset points", color=color)

    color, _, _ = SolveStrategy.get_plot_params(SolveStrategy.CHEN_GREEDY_NOAP)
    ax.annotate("*", xy=results["CHEN_GREEDY_NOAP"][0], xytext=(-10, -6), textcoords="offset points", color=color)
    ax.annotate("**", xy=results["CHEN_GREEDY"][0], xytext=(-15, -6), textcoords="offset points", color=color)

  # plot checkpoint all, TODO: use results from new pickle file
  xlim_min, xlim_max = ax.get_xlim()
  baseline_x = CHECKPOINT_ALL_MEMORY[model]
  color, marker, markersize = SolveStrategy.get_plot_params(SolveStrategy.CHECKPOINT_ALL)
  ax.scatter([baseline_x], [1.0], label="", zorder=2, color=color, marker=marker, s=markersize**2)
  ax.hlines(y=1.0, xmin=xlim_min, xmax=baseline_x, linestyles="dashed", color=color)
  ax.hlines(y=1.0, xmin=baseline_x, xmax=xlim_max, color=color, zorder=2)

  # plot platform mem
  ylim_min, ylim_max = ax.get_ylim()
  mem_gb = platform_memory("p32xlarge") / 1e9
  if xlim_min <= mem_gb <= xlim_max:
      ax.vlines(x=mem_gb, ymin=ylim_min, ymax=ylim_max, linestyles="dotted", color="b")
      ax.set_ylim([ylim_min, ylim_max])
      ax.axvspan(xlim_min, mem_gb, alpha=0.2, color="royalblue")

  tt = ax.text(0.975, 0.975, TITLES[model], fontweight="bold", fontsize=18,
    bbox=dict(facecolor='white', alpha=1.0),
    horizontalalignment="right", verticalalignment="top", transform=ax.transAxes)

  ax.text(0.975, 0.875, INPUT_SHAPES[model], fontsize=14,
    bbox=dict(facecolor='white', alpha=1.0),
    horizontalalignment="right", verticalalignment="top", transform=ax.transAxes)

def valid(r):
  return r is not None and r.schedule_aux_data is not None and r.solve_time_s < 3600

def plot_solver_time(fig, ax):
  result_dict_file = 'data/budget_sweep/p32xlarge_MobileNet_512_None/export_result_dict.pickle'
  result_dict = pickle.load(open(result_dict_file, 'rb'))
  result_dict = {key.name: vals for key, vals in result_dict.items()}

  checkmate_xs, checkmate_ys = \
    zip(*sorted([(r.solver_budget / 1e9, r.solve_time_s) for r in result_dict['OPTIMAL_ILP_GC'] if valid(r)]))
  simrd_xs, simrd_ys = \
    zip(*sorted([(r.solver_budget / 1e9, r.solve_time_s) for r in result_dict['SIMRD'] if r.solver_budget / 1e9 in checkmate_xs]))

  ax.axhline(y=3600, linestyle='dashed', color='maroon', label='ILP timeout (1 hour)')
  color, marker, markersize = SolveStrategy.get_plot_params(SolveStrategy.OPTIMAL_ILP_GC)
  ax.plot(checkmate_xs, checkmate_ys, label='', color=color)
  ax.scatter(checkmate_xs,checkmate_ys, label='', s=markersize**2, c=color, marker=marker)
  color, marker, markersize = SolveStrategy.get_plot_params(SolveStrategy.SIMRD)
  ax.plot(simrd_xs, simrd_ys, label='', color=color)
  ax.scatter(simrd_xs, simrd_ys, s=markersize**2, label='', c=color, marker=marker)
  # ax.set_xlabel('Budget (GB)')
  ax.set_ylabel('Solver Time (s)', fontsize=8)
  ax.set_yscale('log')
  ax.yaxis.tick_right()
  # ax.yaxis.set_label_position('right')
  ax.yaxis.set_tick_params(labelsize=8)
  ax.text(0.99, 0.98, 'MobileNet', fontweight='bold', fontsize=10, \
    bbox=dict(facecolor='white', alpha=1.0, pad=0, edgecolor=None), \
    horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
  # ax.legend(loc='upper right', shadow=False, fancybox=False, framealpha=1.0, fontsize=8)
  # ax.set_title('Budget vs. Solver Time (MobileNet)')
  # plt.savefig('data/budget_sweep/solve_time.png', bbox_inches='tight', dpi=300)

if __name__ == "__main__":
  sns.set(); sns.set_style("white")
  fig, axes = plt.subplots(ncols=3, figsize=(5*3, 4))

  plot_model("VGG16", fig, axes[0])
  plot_model("MobileNet", fig, axes[1])
  plot_model("vgg_unet", fig, axes[2])

  # legend
  legend_items = []
  for strat, label in LEGEND_LABELS.items():
    color, marker, markersize = SolveStrategy.get_plot_params(strat)
    legend_items.append(
      Line2D([0], [0], lw=2, label=label, color=color, marker=marker, markersize=markersize)
    )
  legend_items.append(
    Line2D([0], [0], lw=2, label='ILP timeout (1 hour)', color='maroon', linestyle='dashed')
  )

  ax = fig.add_subplot(111, frameon=False)
  plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
  plt.xlabel('Budget (GB)', fontsize=15, fontweight="bold", labelpad=8)
  plt.ylabel('Overhead (x)', fontsize=15, fontweight="bold", labelpad=8)

  fig.legend(handles=legend_items, loc = 'upper right', bbox_to_anchor = (1.09, 0.9),
            bbox_transform = plt.gcf().transFigure, ncol=1,
            fancybox=False, shadow=False, frameon=True)

  # adaptation text
  plt.text(0.79, 0.01, " * Linearized adaptation", fontsize=10,
    horizontalalignment="right", verticalalignment="top", transform=fig.transFigure)
  plt.text(0.89, 0.01, "** AP adaptation", fontsize=10,
    horizontalalignment="right", verticalalignment="top", transform=fig.transFigure)

  ax = fig.add_subplot(133)
  box = ax.get_position()
  box.x0 = box.x0 + 0.25
  box.x1 = box.x1 + 0.16
  box.y1 = box.y1 - 0.44
  ax.set_position(box)
  plot_solver_time(fig, ax)

  fig.savefig(remat_data_dir() / "budget_sweep" / "figure.png", bbox_inches="tight", dpi=300)
