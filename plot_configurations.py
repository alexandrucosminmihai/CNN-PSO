import pickle
import numpy as np

LOGS_PATH = "./all_logs/toate_logs_2022_02_06-23_13_10.pkl"

with open(LOGS_PATH, mode="rb") as pickle_file:
    logs = pickle.load(pickle_file)


def get_configs_with_attr_value(logs, attr_vals):
    query_result = {
        "query": f"{attr_vals}",
        "configs": [],
    }

    for config_name in logs:
        config = logs[config_name]["configuration"]

        should_be_kept = True
        for attr, val in attr_vals:
            if not (attr in config and config[attr] == val):
                should_be_kept = False
                break

        if should_be_kept is True:
            query_result["configs"].append(config_name)

    return query_result


def get_lines(logs, line_names, line_style, xs_dict_keys, ys_dict_keys):
    lines = []
    for line_name in line_names:
        line = {
            "line_name": line_name,
            "line_style": line_style,
            "xs_dict_keys": [line_name] + xs_dict_keys,
            "ys_dict_keys": [line_name] + ys_dict_keys,
        }
        lines.append(line)

    return lines


def get_particles_space_plot_config(
        logs, search_space_max, num_particles, xs_dict_keys, ys_dict_keys, x_label, y_label
):
    target_configs = get_configs_with_attr_value(
        logs, [("search_space_max", search_space_max), ("num_particles", num_particles)]
    )["configs"]

    return {
        "type": "graph",
        "name": f"Models with particles {num_particles} and search_space_max {search_space_max} {x_label} vs {y_label}",
        "x_label": x_label,
        "y_label": y_label,
        "lines": (
            get_lines(
                logs, line_names=target_configs, line_style="solid",
                xs_dict_keys=xs_dict_keys, ys_dict_keys=ys_dict_keys
            )
        )
    }


def get_particles_space_plot_configs(
        logs, search_space_max_options, num_particles_options, xs_dict_keys, ys_dict_keys, x_label, y_label
):
    plot_configs = []

    for search_space_max in search_space_max_options:
        for num_particles in num_particles_options:
            plot_config = get_particles_space_plot_config(
                logs, search_space_max=search_space_max, num_particles=num_particles,
                xs_dict_keys=xs_dict_keys, ys_dict_keys=ys_dict_keys,
                x_label=x_label, y_label=y_label
            )
            plot_configs.append(plot_config)

    return plot_configs


def sort_models_by_metric(logs, metric_key):  # logs_ys_eval_accuracy_epochs
    candidates = []
    for model_name in logs:
        candidate_best_metric = np.max(logs[model_name][metric_key])
        candidate_best_metric_epoch = np.argmax(logs[model_name][metric_key]) + 1
        candidates.append((candidate_best_metric, candidate_best_metric_epoch, model_name))

    candidates.sort(reverse=True)
    return candidates


print(f"=========================================")
models_sorted_by_acc_particles_10 = []
models_sorted_by_acc_particles_100 = []
models_sorted_by_acc_particles_1000 = []
print(f"Models sorted by best test accuracy:")
print(f"Optimizer\t& Num particles\t& $V_{{max}}$\t& $SearchSpace_{{min}}$\t& $SearchSpace_{{max}}$\t& $\\varphi_1$\t& $\\varphi_2$\t& Test acc\t\\\\")
print(f"\\midrule")
models_sorted_by_acc = sort_models_by_metric(logs, "logs_ys_eval_accuracy_epochs")
for i_model, (acc, best_epoch, model_name) in enumerate(models_sorted_by_acc):
    # print(f"[{i_model + 1}]\t{acc}@e{best_epoch}\t{model_name}")  # Non-Latex
    config = logs[model_name]["configuration"]

    optimizer = config["optimizer"]
    num_particles = config.get("num_particles", "-")
    v_max = config.get("v_max", "-")
    search_space_min = config.get("search_space_min", "-")
    search_space_max = config.get("search_space_max", "-")
    cognitive_coeff = config.get("cognitive_coeff", "-")
    social_coeff = config.get("social_coeff", "-")

    print(f"{optimizer}\t& {num_particles}\t& {v_max}\t& {search_space_min}\t& {search_space_max}\t& {cognitive_coeff}\t& {social_coeff}\t& {acc*100:.2f}\\%\t\\\\")

    if "num_particles" not in config:
        continue
    if config["num_particles"] == 10:
        models_sorted_by_acc_particles_10.append((acc, best_epoch, model_name))
    elif config["num_particles"] == 100:
        models_sorted_by_acc_particles_100.append((acc, best_epoch, model_name))
    elif config["num_particles"] == 1000:
        models_sorted_by_acc_particles_1000.append((acc, best_epoch, model_name))
print(f"\\bottomrule")
print(f"=========================================")
print()

print(f"=========================================")
print(f"10 Particles Models sorted by best test accuracy:")
print(f"Optimizer\t& Num particles\t& $V_{{max}}$\t& $SearchSpace_{{min}}$\t& $SearchSpace_{{max}}$\t& $\\varphi_1$\t& $\\varphi_2$\t& Test acc\t\\\\")
print(f"\\midrule")
for i_model, (acc, best_epoch, model_name) in enumerate(models_sorted_by_acc_particles_10):
    # print(f"[{i_model + 1}]\t{acc}@e{best_epoch}\t{model_name}")
    config = logs[model_name]["configuration"]

    optimizer = config["optimizer"]
    num_particles = config.get("num_particles", "-")
    v_max = config.get("v_max", "-")
    search_space_min = config.get("search_space_min", "-")
    search_space_max = config.get("search_space_max", "-")
    cognitive_coeff = config.get("cognitive_coeff", "-")
    social_coeff = config.get("social_coeff", "-")

    print(f"{optimizer}\t& {num_particles}\t& {v_max}\t& {search_space_min}\t& {search_space_max}\t& {cognitive_coeff}\t& {social_coeff}\t& {acc*100:.2f}\\%\t\\\\")
print(f"\\bottomrule")
print(f"=========================================")
print()

print(f"=========================================")
print(f"100 Particles Models sorted by best test accuracy:")
print(f"Optimizer\t& Num particles\t& $V_{{max}}$\t& $SearchSpace_{{min}}$\t& $SearchSpace_{{max}}$\t& $\\varphi_1$\t& $\\varphi_2$\t& Test acc\t\\\\")
print(f"\\midrule")
for i_model, (acc, best_epoch, model_name) in enumerate(models_sorted_by_acc_particles_100):
    # print(f"[{i_model + 1}]\t{acc}@e{best_epoch}\t{model_name}")
    config = logs[model_name]["configuration"]

    optimizer = config["optimizer"]
    num_particles = config.get("num_particles", "-")
    v_max = config.get("v_max", "-")
    search_space_min = config.get("search_space_min", "-")
    search_space_max = config.get("search_space_max", "-")
    cognitive_coeff = config.get("cognitive_coeff", "-")
    social_coeff = config.get("social_coeff", "-")

    print(f"{optimizer}\t& {num_particles}\t& {v_max}\t& {search_space_min}\t& {search_space_max}\t& {cognitive_coeff}\t& {social_coeff}\t& {acc*100:.2f}\\%\t\\\\")
print(f"\\bottomrule")
print(f"=========================================")
print()

print(f"=========================================")
print(f"1000 Particles Models sorted by best test accuracy:")
print(f"Optimizer\t& Num particles\t& $V_{{max}}$\t& $SearchSpace_{{min}}$\t& $SearchSpace_{{max}}$\t& $\\varphi_1$\t& $\\varphi_2$\t& Test acc\t\\\\")
print(f"\\midrule")
for i_model, (acc, best_epoch, model_name) in enumerate(models_sorted_by_acc_particles_1000):
    # print(f"[{i_model + 1}]\t{acc}@e{best_epoch}\t{model_name}")
    config = logs[model_name]["configuration"]

    optimizer = config["optimizer"]
    num_particles = config.get("num_particles", "-")
    v_max = config.get("v_max", "-")
    search_space_min = config.get("search_space_min", "-")
    search_space_max = config.get("search_space_max", "-")
    cognitive_coeff = config.get("cognitive_coeff", "-")
    social_coeff = config.get("social_coeff", "-")

    print(f"{optimizer}\t& {num_particles}\t& {v_max}\t& {search_space_min}\t& {search_space_max}\t& {cognitive_coeff}\t& {social_coeff}\t& {acc*100:.2f}\\%\t\\\\")
print(f"\\bottomrule")
print(f"=========================================")
print()

models_sorted_by_acc_top3 = (
    models_sorted_by_acc[:2] +
    models_sorted_by_acc_particles_1000[:3] +
    models_sorted_by_acc_particles_100[:3] +
    models_sorted_by_acc_particles_10[:3]
)

print(f"=========================================")
print(f"Top 3 Models for all number of particles sorted by best test accuracy:")
print(f"Optimizer\t& Num particles\t& $V_{{max}}$\t& $SearchSpace_{{min}}$\t& $SearchSpace_{{max}}$\t& $\\varphi_1$\t& $\\varphi_2$\t& Test acc\t\\\\")
print(f"\\midrule")
for i_model, (acc, best_epoch, model_name) in enumerate(models_sorted_by_acc_top3):
    # print(f"[{i_model + 1}]\t{acc}@e{best_epoch}\t{model_name}")
    config = logs[model_name]["configuration"]

    optimizer = config["optimizer"]
    num_particles = config.get("num_particles", "-")
    v_max = config.get("v_max", "-")
    search_space_min = config.get("search_space_min", "-")
    search_space_max = config.get("search_space_max", "-")
    cognitive_coeff = config.get("cognitive_coeff", "-")
    social_coeff = config.get("social_coeff", "-")

    print(f"{optimizer}\t& {num_particles}\t& {v_max}\t& {search_space_min}\t& {search_space_max}\t& {cognitive_coeff}\t& {social_coeff}\t& {acc*100:.2f}\\%\t\\\\")
print(f"\\bottomrule")
print(f"=========================================")
print()

SEARCH_SPACE_MAX_OPTIONS = [0.21, 0.1, 0.005]
NUM_PARTICLES_OPTIONS = [10, 100, 1000]

CONFIGS_GRADIENT = (
        get_configs_with_attr_value(logs, [("optimizer", "sgd")])["configs"]
        + get_configs_with_attr_value(logs, [("optimizer", "adam")])["configs"]
)
CONFIGS_PARTICLES10 = get_configs_with_attr_value(logs, [("num_particles", 10)])["configs"]
CONFIGS_PARTICLES100 = get_configs_with_attr_value(logs, [("num_particles", 100)])["configs"]
CONFIGS_PARTICLES1000 = get_configs_with_attr_value(logs, [("num_particles", 1000)])["configs"]

CONFIGS_PARTICLES10_TOP10 = [model_name for (acc, best_epoch, model_name) in models_sorted_by_acc_particles_10[:10]]
CONFIGS_PARTICLES100_TOP10 = [model_name for (acc, best_epoch, model_name) in models_sorted_by_acc_particles_100[:10]]
CONFIGS_PARTICLES1000_TOP10 = [model_name for (acc, best_epoch, model_name) in models_sorted_by_acc_particles_1000[:10]]

CONFIGS_PARTICLES10_TOP3 = [model_name for (acc, best_epoch, model_name) in models_sorted_by_acc_particles_10[:3]]
CONFIGS_PARTICLES100_TOP3 = [model_name for (acc, best_epoch, model_name) in models_sorted_by_acc_particles_100[:3]]
CONFIGS_PARTICLES1000_TOP3 = [model_name for (acc, best_epoch, model_name) in models_sorted_by_acc_particles_1000[:3]]

CONFIGS_ALL = CONFIGS_GRADIENT + CONFIGS_PARTICLES10 + CONFIGS_PARTICLES100 + CONFIGS_PARTICLES1000


PLOT_CONFIGS_ALL = [
    {
        "type": "graph",
        "name": "All models Epoch vs Test accuracy",
        "x_label": "Num train epochs",
        "y_label": "Test accuracy",
        "lines": (
            get_lines(logs, line_names=CONFIGS_GRADIENT, line_style="solid", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"]) +
            get_lines(logs, line_names=CONFIGS_PARTICLES10, line_style="dotted", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"]) +
            get_lines(logs, line_names=CONFIGS_PARTICLES100, line_style="dashed", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"]) +
            get_lines(logs, line_names=CONFIGS_PARTICLES1000, line_style="dashdot", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"])
        )
    }
]

PLOT_CONFIGS_PARTICLES10 = [
    {
        "type": "graph",
        "name": "All models Epoch vs Test accuracy",
        "x_label": "Num train epochs",
        "y_label": "Test accuracy",
        "lines": (
                get_lines(logs, line_names=CONFIGS_PARTICLES10, line_style="dotted", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"])
        )
    }
]


PLOT_CONFIGS_PARTICLES_SEARCH_SPACE = get_particles_space_plot_configs(
    logs,
    SEARCH_SPACE_MAX_OPTIONS,
    NUM_PARTICLES_OPTIONS,
    xs_dict_keys=["logs_xs_epochs"],
    ys_dict_keys=["logs_ys_eval_accuracy_epochs"],
    x_label="Train epochs",
    y_label="Test accuracy"
)

PLOT_CONFIGS_PARTICLES_TOP10 = [
    {
        "type": "graph",
        "name": "Top 10 10Particles models Epoch vs Test accuracy",
        "x_label": "Num train epochs",
        "y_label": "Test accuracy",
        "lines": (
            get_lines(logs, line_names=CONFIGS_GRADIENT, line_style="solid", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"]) +
            get_lines(logs, line_names=CONFIGS_PARTICLES10_TOP10, line_style="dotted", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"])
        )
    },
    {
        "type": "graph",
        "name": "Top 10 100Particles models Epoch vs Test accuracy",
        "x_label": "Num train epochs",
        "y_label": "Test accuracy",
        "lines": (
            get_lines(logs, line_names=CONFIGS_GRADIENT, line_style="solid", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"]) +
            get_lines(logs, line_names=CONFIGS_PARTICLES100_TOP10, line_style="dashed", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"])
        )
    },
    {
        "type": "graph",
        "name": "Top 10 1000Particles models Epoch vs Test accuracy",
        "x_label": "Num train epochs",
        "y_label": "Test accuracy",
        "lines": (
            get_lines(logs, line_names=CONFIGS_GRADIENT, line_style="solid", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"]) +
            get_lines(logs, line_names=CONFIGS_PARTICLES1000_TOP10, line_style="dashdot", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"])
        )
    }
]

PLOT_CONFIGS_PARTICLES_TOP3 = [
    {
        "type": "graph",
        "name": "Top 3 from each num Particles models Epoch vs Test accuracy",
        "x_label": "Num train epochs",
        "y_label": "Test accuracy",
        "lines": (
            get_lines(logs, line_names=CONFIGS_GRADIENT, line_style="solid", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"]) +
            get_lines(logs, line_names=CONFIGS_PARTICLES10_TOP10[:3], line_style="dotted", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"]) +
            get_lines(logs, line_names=CONFIGS_PARTICLES100_TOP10[:3], line_style="dashed", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"]) +
            get_lines(logs, line_names=CONFIGS_PARTICLES1000_TOP10[:3], line_style="dashdot", xs_dict_keys=["logs_xs_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"])
        )
    },

    {
        "type": "graph",
        "name": "Top 3 from each num Particles models Training duration vs Test accuracy",
        "x_label": "Training duration (s)",
        "y_label": "Test accuracy",
        "lines": (
            get_lines(logs, line_names=CONFIGS_GRADIENT, line_style="solid", xs_dict_keys=["logs_ys_durations_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"]) +
            get_lines(logs, line_names=CONFIGS_PARTICLES10_TOP10[:3], line_style="dotted", xs_dict_keys=["logs_xs_durations_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"]) +
            get_lines(logs, line_names=CONFIGS_PARTICLES100_TOP10[:3], line_style="dashed", xs_dict_keys=["logs_xs_durations_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"]) +
            get_lines(logs, line_names=CONFIGS_PARTICLES1000_TOP10[:3], line_style="dashdot", xs_dict_keys=["logs_xs_durations_epochs"], ys_dict_keys=["logs_ys_eval_accuracy_epochs"])
        )
    }
]
