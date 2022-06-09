import os
import argparse
import pickle
from datetime import datetime
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
from torch.optim import Adam

from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader

from utils import set_seed, plot_all_graphs
from models import Model1, PSOModel

from train_configurations import TRAIN_CONFIGURATIONS


def append_line_logs(
    all_logs_tuple,
    line_name,
    logs_xs_epochs, logs_ys_losses_epochs, logs_ys_durations_epochs,
    logs_xs_eval_epochs, logs_ys_eval_losses_epochs, logs_ys_eval_accuracy_epochs
):
    (
        all_logs_xs_epochs, all_logs_ys_losses_epochs, all_logs_ys_durations_epochs,
        all_logs_xs_eval_epochs, all_logs_ys_eval_losses_epochs, all_logs_ys_eval_accuracy_epochs,
        line_names
    ) = all_logs_tuple

    all_logs_xs_epochs.append(logs_xs_epochs)
    all_logs_ys_losses_epochs.append(logs_ys_losses_epochs)
    all_logs_ys_durations_epochs.append(logs_ys_durations_epochs)
    all_logs_xs_eval_epochs.append(logs_xs_eval_epochs)
    all_logs_ys_eval_losses_epochs.append(logs_ys_eval_losses_epochs)
    all_logs_ys_eval_accuracy_epochs.append(logs_ys_eval_accuracy_epochs)
    line_names.append(line_name)


def train_pso_model(
    num_epochs,
    conv_output_size, image_channels,
    num_particles,
    search_space,
    v_max, inertia_weight,
    cognitive_coeff, social_coeff
):
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    model = PSOModel(
        model_class=Model1,
        model_params={
            "conv_output_size": conv_output_size,
            "image_channels": image_channels,
        },
        num_particles=num_particles,
        v_max=v_max,
        search_space=search_space,
        inertia_weight=inertia_weight,
        cognitive_coeff=cognitive_coeff,
        social_coeff=social_coeff,
    )
    model_name = (
        f"pso_epochs{num_epochs}_particles{num_particles}_vmax{v_max:.6f}_"
        f"space{search_space[0]:.6f}-{search_space[1]:.6f}_inertia{inertia_weight}_"
        f"cognitive{cognitive_coeff:.3f}_social{social_coeff:.3f}"
    )
    model_name_timestamped = f"{model_name}_{timestamp}"
    model_savedir = f"./saved_models/{model_name_timestamped}"
    Path(model_savedir).mkdir(parents=True, exist_ok=True)

    (
        logs_xs_epochs, logs_ys_losses_epochs, logs_ys_durations_epochs,
        logs_xs_eval_epochs, logs_ys_eval_losses_epochs, logs_ys_eval_accuracy_epochs
    ) = model.train_model(data_loaders, num_epochs)
    model_savedata_dict = {
        "logs_xs_epochs": logs_xs_epochs,
        "logs_ys_losses_epochs": logs_ys_losses_epochs,
        "logs_ys_durations_epochs": logs_ys_durations_epochs,
        "logs_xs_eval_epochs": logs_xs_eval_epochs,
        "logs_ys_eval_losses_epochs": logs_ys_eval_losses_epochs,
        "logs_ys_eval_accuracy_epochs": logs_ys_eval_accuracy_epochs,
        "model_name": model_name,
        "timestamp": timestamp,
    }

    # Save the logs as a pickle file.
    logs_file_path = f"{model_savedir}/logs_{model_name_timestamped}.pkl"
    with open(logs_file_path, "wb") as pickle_file:
        pickle.dump(model_savedata_dict, pickle_file)
    print(f"Saved the model's logs to {logs_file_path}.")

    # Save the model weights.
    weights_file_path = f"{model_savedir}/weights_{model_name_timestamped}.pt"
    torch.save(model.state_dict(), weights_file_path)
    print(f"Saved the model's weights to {weights_file_path}.")


def get_model_name(train_config):
    if train_config["optimizer"] == "pso":
        model_name = (
            f"{train_config['optimizer']}_e{train_config['num_epochs']}_parts{train_config['num_particles']}_vmax{train_config['v_max']:.6f}_"
            f"space{train_config['search_space_min']:.6f}-{train_config['search_space_max']:.6f}_inertia{train_config['inertia_weight']}_"
            f"cognitive{train_config['cognitive_coeff']:.3f}_social{train_config['social_coeff']:.3f}"
        )
    elif train_config["optimizer"] == "sgd":
        model_name = f"{train_config['optimizer']}_epochs{train_config['num_epochs']}_lr{train_config['learning_rate']}"
    elif train_config["optimizer"] == "adam":
        model_name = f"{train_config['optimizer']}_epochs{train_config['num_epochs']}_lr{train_config['learning_rate']}"
    else:
        raise ValueError(f"Unknown optimizer '{train_config['optimizer']}'.")

    return model_name


NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # Load the MNIST dataset.
    train_set = datasets.MNIST(
        root="data",
        train=True,
        transform=Compose([
            ToTensor(),
            Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]),
        download=True
    )
    test_set = datasets.MNIST(
        root="data",
        train=False,
        transform=Compose([
            ToTensor(),
            Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]),
    )
    print(f"[INFO] train_set.data.shape = {train_set.data.shape}")
    print(f"[INFO] test_set.data.shape = {test_set.data.shape}")
    conv_output_size = 7
    image_channels = 1

    all_logs = dict()

    print(f"[INFO] A total of {len(TRAIN_CONFIGURATIONS)} configurations will be trained.")
    print()

    for train_config in TRAIN_CONFIGURATIONS:
        model_name = get_model_name(train_config)
        print(f"[STATUS] Starting to run the train configuration for model {model_name}.")
        pprint(train_config)

        # Prepare the dataset to be loaded in batches.
        data_loaders = {
            "train": torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=len(train_set),
                shuffle=False,
                num_workers=1,
            ),
            "test": torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=50,
                shuffle=False,
                num_workers=1,
            ),
        }

        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        if train_config["optimizer"] == "pso":
            model = PSOModel(
                model_class=Model1,
                model_params={
                    "conv_output_size": conv_output_size,
                    "image_channels": image_channels,
                },
                num_particles=train_config["num_particles"],
                v_max=train_config["v_max"],
                search_space=(train_config["search_space_min"], train_config["search_space_max"]),
                inertia_weight=train_config["inertia_weight"],
                cognitive_coeff=train_config["cognitive_coeff"],
                social_coeff=train_config["social_coeff"],
            )
            optimizer = None
        elif train_config["optimizer"] == "sgd":
            model = Model1(conv_output_size=conv_output_size, image_channels=image_channels)
            optimizer = torch.optim.SGD(model.parameters(), lr=train_config["learning_rate"])  # lr=0.1
        elif train_config["optimizer"] == "adam":
            model = Model1(conv_output_size=conv_output_size, image_channels=image_channels)
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])  # lr=0.001
        else:
            raise ValueError(f"Unknown optimizer '{train_config['optimizer']}'.")

        model_name_timestamped = f"{model_name}_{timestamp}"
        model_savedir = f"./saved_models/{model_name_timestamped}"
        Path(model_savedir).mkdir(parents=True, exist_ok=True)

        (
            logs_xs_epochs, logs_ys_losses_epochs, logs_ys_durations_epochs,
            logs_xs_eval_epochs, logs_ys_eval_losses_epochs, logs_ys_eval_accuracy_epochs
        ) = model.train_model(data_loaders, train_config['num_epochs'], optimizer)
        model_savedata_dict = {
            "logs_xs_epochs": logs_xs_epochs,
            "logs_ys_losses_epochs": logs_ys_losses_epochs,
            "logs_ys_durations_epochs": logs_ys_durations_epochs,
            "logs_xs_eval_epochs": logs_xs_eval_epochs,
            "logs_ys_eval_losses_epochs": logs_ys_eval_losses_epochs,
            "logs_ys_eval_accuracy_epochs": logs_ys_eval_accuracy_epochs,
            "model_name": model_name,
            "timestamp": timestamp,
            "configuration": train_config,
        }

        # Save the logs as a pickle file.
        logs_file_path = f"{model_savedir}/logs_{model_name_timestamped}.pkl"
        with open(logs_file_path, "wb") as pickle_file:
            pickle.dump(model_savedata_dict, pickle_file)
        print(f"[STATUS] Saved the model's logs to {logs_file_path}.")

        # Save the model weights.
        weights_file_path = f"{model_savedir}/weights_{model_name_timestamped}.pt"
        torch.save(model.state_dict(), weights_file_path)
        print(f"[STATUS] Saved the model's weights to {weights_file_path}.")

        # Add the current train configuration logs to the central logs dict.
        all_logs[model_name] = model_savedata_dict

        if train_config["optimizer"] != "pso":
            serialized_params = nn.utils.parameters_to_vector(model.parameters())
            print(f"[STATUS] Statistics for the weights of the model trained with {train_config['optimizer']}, after {train_config['num_epochs']}:")
            print(f"  > min weight: {torch.min(serialized_params).item()}")
            print(f"  > max weight: {torch.max(serialized_params).item()}")
            print(f"  > mean weight: {torch.mean(serialized_params).item()}")
            print(f"  > median weight: {torch.median(serialized_params).item()}")
            print()

        print()

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    all_logs_file_path = f"./all_logs/logs_{timestamp}.pkl"
    Path(all_logs_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(all_logs_file_path, mode="wb") as logs_pickle:
        pickle.dump(all_logs, logs_pickle)

    print(f"[STATUS] Saved all the train configurations logs in a central dictionary at {all_logs_file_path}.")
