import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sn
import pandas as pd


from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Tuple, List

FIGS_PATH = "./figs"


def set_seed(seed: int = 42):
    """
    Sets a seed to ensure reproducibility
    """

    # Torch seeds.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Numpy and Random seeds.
    np.random.seed(seed)
    random.seed(seed)


def get_graph_file_name(graph_name, use_timestamp=True):
    file_name = f"{'_'.join(graph_name.split())}"
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        file_name += "_" + timestamp
    file_name += ".png"

    return file_name


@torch.no_grad()
def plot_graph(
        lines: List[Tuple[List, List]],
        line_styles: List[str] = None,
        standard_deviations_per_lines: List[Tuple[List, List]] = None,
        line_names: List[str] = None,
        graph_name: str = None, x_label: str = None, y_label: str = None,
        figsize=(16, 9),
        save_to_file=True, savedir_path=FIGS_PATH, use_timestamp=True
):
    plt.figure(num=1, figsize=figsize)

    num_lines = len(lines)
    if line_styles is None:
        line_styles = ["solid" for _ in range(num_lines)]

    if standard_deviations_per_lines is None:
        for (xs_line, ys_line), line_name, line_style in zip(lines, line_names, line_styles):
            for i_x in range(len(xs_line)):
                if isinstance(xs_line[i_x], torch.Tensor):
                    xs_line[i_x] = xs_line[i_x].cpu()
            for i_y in range(len(ys_line)):
                if isinstance(ys_line[i_y], torch.Tensor):
                    ys_line[i_y] = ys_line[i_y].cpu()
            plt.plot(xs_line, ys_line, label=line_name, alpha=0.7, linestyle=line_style)
    else:
        for (xs_line, ys_line), line_name, (_, standard_deviations), line_style in zip(
                lines, line_names, standard_deviations_per_lines, line_styles
        ):
            if isinstance(xs_line, torch.Tensor):
                xs_line = xs_line.cpu()
            if isinstance(ys_line, torch.Tensor):
                ys_line = ys_line.cpu()
            plt.plot(xs_line, ys_line, label=line_name, alpha=0.7, linestyle=line_style)
            ys_line = np.array(ys_line)
            standard_deviations = np.array(standard_deviations)
            plt.fill_between(xs_line, ys_line - standard_deviations, ys_line + standard_deviations, alpha=0.4)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(graph_name)

    if save_to_file:
        file_name = get_graph_file_name(graph_name, use_timestamp)
        Path(savedir_path).mkdir(parents=True, exist_ok=True)
        file_path_str = os.path.join(savedir_path, file_name)

        plt.tight_layout()
        plt.savefig(file_path_str)

    plt.clf()


def check_num_lines(l, l_name, expected_len):
    if len(l) != expected_len:
        print(f"{l_name} does not contain data for all {expected_len} optimizers! It contains only {len(l)} lines.")
        return False
    return True


def check_num_points(lines, lines_name=None):
    num_points_dict = defaultdict(lambda: 0)

    for line in lines:
        num_points = len(line)
        num_points_dict[num_points] += 1

    if len(num_points_dict) != 1:
        print(f"Not all lines {'in ' + lines_name + ' ' if lines_name else ''} have the same number of points!")
        for num_points in num_points_dict:
            print(f"- {num_points_dict[num_points]} have {num_points} points")
        return False

    return True


def plot_one_graph(all_lines_xs, all_lines_ys, line_names, line_styles, graph_name, x_label, y_label, savedir_path):
    lines = []
    for (line_xs_list, line_ys_list) in zip(all_lines_xs, all_lines_ys):
        lines.append((line_xs_list, line_ys_list))

    plot_graph(
        lines=lines,
        standard_deviations_per_lines=None,
        line_names=line_names,
        line_styles=line_styles,
        graph_name=graph_name,
        x_label=x_label,
        y_label=y_label,
        savedir_path=savedir_path,
    )


@torch.no_grad()
def plot_confusion_matrix(cm, graph_name, class_names, savedir_path, use_timestamp=True, figsize=(20, 20), normalize_rows=True):
    if normalize_rows:
        cm = np.asarray(cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, range(len(class_names)), range(len(class_names)))
    plt.figure(num=1, figsize=figsize)
    sn.set(font_scale=1) # for label size
    if normalize_rows is False:
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="d", cbar=False, xticklabels=class_names, yticklabels=class_names) # font size
    else:
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=".2f", cbar=False, xticklabels=class_names, yticklabels=class_names) # font size

    file_name = get_graph_file_name(graph_name, use_timestamp)
    Path(savedir_path).mkdir(parents=True, exist_ok=True)
    file_path_str = os.path.join(savedir_path, file_name)

    plt.title(graph_name)
    # plt.xlabel("True labels")
    # plt.ylabel("Predicted labels")
    # plt.xticks(ticks=plt.gca().get_xticks(), labels=class_names, fontsize=8)
    # plt.yticks(ticks=plt.gca().get_yticks(), labels=class_names, fontsize=8)
    plt.tight_layout()

    plt.savefig(file_path_str)

    plt.clf()


def plot_all_graphs(
    all_logs_xs_epochs, all_logs_ys_losses_epochs, all_logs_ys_durations_epochs,
    all_logs_xs_eval_epochs, all_logs_ys_eval_losses_epochs, all_logs_ys_eval_accuracy_epochs,
    line_names,
):
    """
    Each param represents a list of line-level elements, more precisely each param is a list of lists.

    Plots:
    1. Train loss vs epochs.
    2. Train loss vs duration.
    3. Test loss vs epochs.
    4. Test loss vs duration.
    5. Test accuracy vs epochs.
    6. Test accuracy vs duration.
    """
    num_lines = len(line_names)
    num_lines_check = (
        check_num_lines(all_logs_xs_epochs, "all_logs_xs_epochs", num_lines) and
        check_num_lines(all_logs_ys_losses_epochs, "all_logs_ys_losses_epochs", num_lines) and
        check_num_lines(all_logs_ys_durations_epochs, "all_logs_ys_durations_epochs", num_lines) and
        check_num_lines(all_logs_xs_eval_epochs, "all_logs_xs_eval_epochs", num_lines) and
        check_num_lines(all_logs_ys_eval_losses_epochs, "all_logs_ys_eval_losses_epochs", num_lines) and
        check_num_lines(all_logs_ys_eval_accuracy_epochs, "all_logs_ys_eval_accuracy_epochs", num_lines)
    )
    if not num_lines_check:
        return

    num_points_check = (
        check_num_points(all_logs_xs_epochs, "all_logs_xs_epochs") and
        check_num_points(all_logs_ys_losses_epochs, "all_logs_ys_losses_epochs") and
        check_num_points(all_logs_ys_durations_epochs, "all_logs_ys_durations_epochs") and
        check_num_points(all_logs_xs_eval_epochs, "all_logs_xs_eval_epochs") and
        check_num_points(all_logs_ys_eval_losses_epochs, "all_logs_ys_eval_losses_epochs") and
        check_num_points(all_logs_ys_eval_accuracy_epochs, "all_logs_ys_eval_accuracy_epochs")
    )
    if not num_points_check:
        return

    all_logs_xs_durations_epochs = []
    for logs_ys_durations_steps in all_logs_ys_durations_epochs:
        logs_xs_duration = []
        for i_epoch, epoch_duration in enumerate(logs_ys_durations_steps):
            if i_epoch == 0:
                logs_xs_duration.append(epoch_duration)
            else:
                logs_xs_duration.append(epoch_duration + logs_xs_duration[-1])
        all_logs_xs_durations_epochs.append(logs_xs_duration)

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    savedir_path = f"./figs/{timestamp}"

    # 1. Train loss vs epochs.
    plot_one_graph(
        all_logs_xs_epochs, all_logs_ys_losses_epochs,
        line_names=line_names, graph_name="Train Loss vs Epochs",
        x_label="Number of epochs", y_label="Training loss",
        savedir_path=savedir_path
    )

    # 2. Train loss vs duration.
    plot_one_graph(
        all_logs_xs_durations_epochs, all_logs_ys_losses_epochs,
        line_names=line_names, graph_name="Train Loss vs Training time (s)",
        x_label="Training time (s)", y_label="Training loss",
        savedir_path=savedir_path
    )

    # 3. Test loss vs epochs.
    plot_one_graph(
        all_logs_xs_eval_epochs, all_logs_ys_eval_losses_epochs,
        line_names=line_names, graph_name="Test Loss vs Epochs",
        x_label="Number of epochs", y_label="Test loss",
        savedir_path=savedir_path
    )

    # 4. Test loss vs duration.
    plot_one_graph(
        all_logs_xs_durations_epochs, all_logs_ys_eval_losses_epochs,
        line_names=line_names, graph_name="Test Loss vs Training time (s)",
        x_label="Training time (s)", y_label="Test loss",
        savedir_path=savedir_path
    )

    # 5. Test accuracy vs epochs.
    plot_one_graph(
        all_logs_xs_eval_epochs, all_logs_ys_eval_accuracy_epochs,
        line_names=line_names, graph_name="Test Accuracy vs Epochs",
        x_label="Number of epochs", y_label="Test Accuracy",
        savedir_path=savedir_path
    )

    # 6. Test accuracy vs duration.
    plot_one_graph(
        all_logs_xs_durations_epochs, all_logs_ys_eval_accuracy_epochs,
        line_names=line_names, graph_name="Test Accuracy vs Training time (s)",
        x_label="Training time (s)", y_label="Test Accuracy",
        savedir_path=savedir_path
    )
