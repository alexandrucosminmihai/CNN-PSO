import pickle
from pathlib import Path
from datetime import datetime

# from configurations import PLOT_CONFIGURATIONS_UPB, PLOT_CONFIGURATIONS_BDD10K, PLOT_CONFIGURATIONS_PASCALVOC, PLOT_CONFIGURATIONS_CITYSCAPES
from plot_configurations import PLOT_CONFIGS_ALL, PLOT_CONFIGS_PARTICLES10, PLOT_CONFIGS_PARTICLES_SEARCH_SPACE, PLOT_CONFIGS_PARTICLES_TOP10, PLOT_CONFIGS_PARTICLES_TOP3
from utils import plot_one_graph, plot_confusion_matrix

LOGS_PATH = "./all_logs/toate_logs_2022_02_06-23_13_10.pkl"
FIGS_ROOT = "./figs"

PLOT_CONFIGURATIONS = PLOT_CONFIGS_PARTICLES_SEARCH_SPACE + PLOT_CONFIGS_PARTICLES_TOP10 + PLOT_CONFIGS_PARTICLES_TOP3

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    logs_path = Path(LOGS_PATH)
    if not logs_path.exists():
        raise AssertionError(f"Logs file does not exist at {LOGS_PATH}")
    logs = None
    with open(LOGS_PATH, mode="rb") as logs_file:
        logs = pickle.load(logs_file)

    for model_name in logs:
        model_dict = logs[model_name]
        num_epochs = len(model_dict["logs_ys_durations_epochs"])

        model_dict["logs_xs_durations_epochs"] = []
        for i_epoch in range(num_epochs):
            model_dict["logs_xs_durations_epochs"].append(sum(model_dict["logs_ys_durations_epochs"][:(i_epoch + 1)]))

total_training_duration = 0.0
for model_name in logs:
    model_dict = logs[model_name]
    training_duration = sum(model_dict["logs_ys_durations_epochs"])
    total_training_duration += training_duration

    figs_dir_name = f"{timestamp}_generated_from_{logs_path.name}"
    figs_dir_path = f"{FIGS_ROOT}/{figs_dir_name}"

    for plot_config in PLOT_CONFIGURATIONS:
        if plot_config["type"] == "graph":
            graph_name = plot_config["name"]
            x_label = plot_config["x_label"]
            y_label = plot_config["y_label"]

            all_lines_xs = []
            all_lines_ys = []
            line_names = []
            line_styles = []

            for line_info in plot_config["lines"]:
                line_name = line_info["line_name"]
                line_style = line_info["line_style"]
                xs_dict_keys = line_info["xs_dict_keys"]
                ys_dict_keys = line_info["ys_dict_keys"]

                xs = logs
                for dict_key in xs_dict_keys:
                    xs = xs[dict_key]

                ys = logs
                for dict_key in ys_dict_keys:
                    ys = ys[dict_key]

                line_names.append(line_name)
                line_styles.append(line_style)
                all_lines_xs.append(xs)
                all_lines_ys.append(ys)

            plot_one_graph(
                all_lines_xs, all_lines_ys,
                line_names, line_styles,
                graph_name,
                x_label, y_label,
                savedir_path=figs_dir_path
            )
        elif plot_config["type"] == "confusion_matrix":
            class_names = plot_config["class_names"]

            cm = logs
            for dict_key in plot_config["cm_dict_keys"]:
                cm = cm[dict_key]

            epoch_index = logs
            for dict_key in plot_config["epoch_index_dict_keys"]:
                epoch_index = epoch_index[dict_key]

            epoch_val_miou = logs
            for dict_key in plot_config["epoch_val_miou_dict_keys"]:
                epoch_val_miou = epoch_val_miou[dict_key]

            graph_name = plot_config["name"]
            if graph_name is None:
                graph_name = ""
            graph_name = f"{graph_name} Epoch {epoch_index} Val mIOU {epoch_val_miou:.3f}"

            plot_confusion_matrix(cm, graph_name, class_names, savedir_path=figs_dir_path)
        else:
            raise AssertionError(f"Unknown plot type '{plot_config['type']}'.")
