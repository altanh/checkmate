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
from remat.tensorflow2.extraction import dfgraph_from_keras

from remat.core.solvers.strategy_simrd import solve_simrd
from simrd.heuristic import DTR, DTREqClass, DTRLocal, MSPS, LRU, LargestStorage, RandomStorage
from simrd.runtime import RuntimeV2EagerOptimized

NUM_ILP_CORES = os.environ.get("ILP_CORES", 12 if os.cpu_count() > 12 else 4)

MODEL_PLATFORM = {
    'VGG16': 'p32xlarge',
    'vgg_unet': 'p32xlarge',
    'MobileNet': 'p32xlarge',
    # ...
}

MODEL_BATCH = {
    'VGG16': '256',
    'vgg_unet': '32',
    'MobileNet': '512',
    # ...
}

MODEL_KEY = {
    'VGG16': 'p32xlarge_VGG16_256_None',
    'vgg_unet': 'p32xlarge_vgg_unet_32_None',
    'MobileNet': 'p32xlarge_MobileNet_512_None',
    # ...
}

MODEL_INPUT_SHAPE = {
    'VGG16': '224x224',
    'vgg_unet': '416x608',
    'MobileNet': '224x224',
    # ...
}

PLOT_UNIT_RAM = 1e9

# def extract_params():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--platform', default="flops", choices=PLATFORM_CHOICES)
#     parser.add_argument('--model-name', default="VGG16", choices=list(sorted(MODEL_NAMES)))
#     parser.add_argument("-b", "--batch-size", type=int, default=1)
#     parser.add_argument("-s", "--input-shape", type=int, nargs="+", default=[])

#     _args = parser.parse_args()
#     _args.input_shape = _args.input_shape if _args.input_shape else None
#     return _args


def prefix_min_np(values: np.ndarray):
    assert values.ndim == 1
    values_min = np.copy(values)
    for i in range(1, values.shape[0]):
        values_min[i] = min(values_min[i - 1], values[i])
    return values_min


def plot_strategy(ax, results, color, marker, markersize, baseline_cpu, zorder):
    valid_data = [r.schedule_aux_data for r in results if r is not None and r.schedule_aux_data is not None]
    sorted_data = sorted(valid_data, key=lambda r: r.peak_ram)
    data_points = [(t.peak_ram / PLOT_UNIT_RAM, t.cpu * 1.0 / baseline_cpu) for t in sorted_data]
    if not len(data_points):
        return

    x, y = map(list, zip(*data_points))
    x_step, y_step = x + [10000], prefix_min_np(np.array(y + [min(y)]))
    ax.step(x_step, y_step, where='post', zorder=1, color=color)
    # Plot only the first and last points
    ax.scatter([x[0], x[-1]], [y[0], y[-1]], label="", zorder=zorder, s=markersize ** 2,
                color=color, marker=marker, alpha=0.75)

def plot_model(model_name, fig, ax):
    log_base = remat_data_dir() / 'budget_sweep' / MODEL_KEY[model_name]
    result_dict = pickle.load((log_base / 'result_dict.pickle').open('rb'))
    simrd_results = pickle.load((log_base / 'simrd_results.pickle').open('rb'))
    simrd_heuristics = pickle.load((log_base / 'simrd_heuristics.pickle').open('rb'))
    baseline_cpu = result_dict[SolveStrategy.CHECKPOINT_ALL][0].schedule_aux_data.cpu
    baseline_memory = result_dict[SolveStrategy.CHECKPOINT_ALL][0].schedule_aux_data.peak_ram

    for solve_strategy, results in result_dict.items():
        if solve_strategy in [SolveStrategy.CHECKPOINT_LAST_NODE, SolveStrategy.CHECKPOINT_ALL]: continue
        color, marker, markersize = SolveStrategy.get_plot_params(solve_strategy)
        scatter_zorder = 3 if solve_strategy == SolveStrategy.CHECKPOINT_ALL_AP else 2
        plot_strategy(ax, results, color, marker, markersize, baseline_cpu, scatter_zorder)

    # Plot simrd results
    for heuristic, results in zip(simrd_heuristics, simrd_results):
        color, marker, markersize = heuristic.COLOR, heuristic.MARKER, matplotlib.rcParams['lines.markersize']
        plot_strategy(ax, results, color, marker, markersize, baseline_cpu, 2)

    # Plot ideal (checkpoint all)
    xlim_min, xlim_max = ax.get_xlim()
    checkpoint_all_result = result_dict[SolveStrategy.CHECKPOINT_ALL][0].schedule_aux_data
    x = baseline_memory / PLOT_UNIT_RAM
    y = 1.0
    color, marker, markersize = SolveStrategy.get_plot_params(SolveStrategy.CHECKPOINT_ALL)
    xlim_max = max(x, xlim_max)
    ax.scatter([x], [y], label="", zorder=2, color=color, marker=marker, s=markersize ** 2)
    ax.hlines(y=y, xmin=xlim_min, xmax=x, linestyles="dashed", color=color)
    ax.hlines(y=y, xmin=x, xmax=xlim_max, color=color, zorder=2)
    ax.set_xlim([xlim_min, xlim_max])

    # Plot platform memory
    ylim_min, ylim_max = ax.get_ylim()
    mem_gb = platform_memory(MODEL_PLATFORM[model_name]) / 1e9
    if xlim_min <= mem_gb <= xlim_max:
        ax.vlines(x=mem_gb, ymin=ylim_min, ymax=ylim_max, linestyles="dotted", color="b")
        ax.set_ylim([ylim_min, ylim_max])
        ax.axvspan(xlim_min, mem_gb, alpha=0.2, color='royalblue')

if __name__ == "__main__":
    logger = logging.getLogger("budget_sweep")
    logger.setLevel(logging.DEBUG)
    log_base = remat_data_dir() / 'budget_sweep'

    sns.set()
    sns.set_style('white')

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_model('MobileNet', fig, ax)
    ax.set_xlim([4,48])
    ax.set_ylim([0.95, 1.5])

    fig.savefig(log_base / 'the_figure3.png', bbox_inches='tight', dpi=300)
