def generate_configurations(
        optimizer_options,
        num_particles_options,
        search_space_options,
        v_max_options,
        inertia_weight_options,
        cognitive_social_coeff_options,
        num_epochs
):
    configs = []

    for optimizer in optimizer_options:
        for num_particles in num_particles_options:
            for search_space in search_space_options:
                search_space_min, search_space_max = search_space

                for v_max in v_max_options:
                    if isinstance(v_max, str):
                        space_dimension_length = search_space_max - search_space_min

                        if v_max == "full":
                            v_max = space_dimension_length
                        elif v_max == "half":
                            v_max = space_dimension_length / 2
                        elif v_max == "quarter":
                            v_max = space_dimension_length / 4
                        else:
                            raise ValueError(f"Unknown v_max configuration '{v_max}'.")

                    for inertia_weight in inertia_weight_options:
                        for cognitive_social_coeff in cognitive_social_coeff_options:
                            cognitive_coeff, social_coeff = cognitive_social_coeff

                            configs.append(
                                {
                                    "optimizer": optimizer,

                                    "num_particles": num_particles,
                                    "v_max": v_max,
                                    "search_space_min": search_space_min,
                                    "search_space_max": search_space_max,
                                    "inertia_weight": inertia_weight,
                                    "cognitive_coeff": cognitive_coeff,
                                    "social_coeff": social_coeff,

                                    "num_epochs": num_epochs,
                                }
                            )

    return configs


NUM_EPOCHS = 10

SGD_CONFIGURATION = {
    "optimizer": "sgd",

    "learning_rate": 0.1,

    "num_epochs": NUM_EPOCHS,
}

ADAM_CONFIGURATION = {
    "optimizer": "adam",

    "learning_rate": 0.001,

    "num_epochs": NUM_EPOCHS,
}

ORIGINAL_CONFIGURATIONS = [
    {
        "optimizer": "pso",

        "num_particles": 1000,
        "v_max": 0.005,
        "search_space_min":-0.005,
        "search_space_max":0.005,
        "inertia_weight": 0.8,
        "cognitive_coeff": 0.1,
        "social_coeff": 0.1,

        "num_epochs": NUM_EPOCHS,
    },

    {
        "optimizer": "pso",

        "num_particles": 1000,
        "v_max": 0.1,
        "search_space_min": -0.2,
        "search_space_max": 0.2,
        "inertia_weight": 0.8,
        "cognitive_coeff": 0.1,
        "social_coeff": 0.1,

        "num_epochs": NUM_EPOCHS,
    },

    {
        "optimizer": "pso",

        "num_particles": 1000,
        "v_max": 0.1,
        "search_space_min": -0.1,
        "search_space_max": 0.1,
        "inertia_weight": 0.8,
        "cognitive_coeff": 0.1,
        "social_coeff": 0.1,

        "num_epochs": NUM_EPOCHS,
    },

    {
        "optimizer": "pso",

        "num_particles": 100,
        "v_max": 0.1,
        "search_space_min": -0.1,
        "search_space_max": 0.1,
        "inertia_weight": 0.8,
        "cognitive_coeff": 0.1,
        "social_coeff": 0.1,

        "num_epochs": NUM_EPOCHS,
    },
]

QUICK_CONFIGURATIONS = [
    {
        "optimizer": "pso",

        "num_particles": 10,
        "v_max": 0.005,
        "search_space_min":-0.005,
        "search_space_max":0.005,
        "inertia_weight": 0.8,
        "cognitive_coeff": 0.1,
        "social_coeff": 0.1,

        "num_epochs": NUM_EPOCHS,
    },

    {
        "optimizer": "pso",

        "num_particles": 10,
        "v_max": 0.1,
        "search_space_min": -0.2,
        "search_space_max": 0.2,
        "inertia_weight": 0.8,
        "cognitive_coeff": 0.1,
        "social_coeff": 0.1,

        "num_epochs": NUM_EPOCHS,
    },

    {
        "optimizer": "pso",

        "num_particles": 10,
        "v_max": 0.1,
        "search_space_min": -0.1,
        "search_space_max": 0.1,
        "inertia_weight": 0.8,
        "cognitive_coeff": 0.1,
        "social_coeff": 0.1,

        "num_epochs": NUM_EPOCHS,
    },
]

GRID_CONFIGURATIONS = generate_configurations(
    optimizer_options=["pso"],
    num_particles_options=[10, 100, 1000],
    search_space_options=[
        (-0.21, 0.21),
        (-0.1, 0.1),
        (-0.005, 0.005),
    ],
    v_max_options=["full", "half", "quarter"],
    inertia_weight_options=[0.75],
    cognitive_social_coeff_options=[
        (0.1, 0.1),
        (1, 1),
        (1, 2),
        (2, 1)
    ],
    num_epochs=10
)

# TRAIN_CONFIGURATIONS = [
#     *ORIGINAL_CONFIGURATIONS,
#     SGD_CONFIGURATION,
#     ADAM_CONFIGURATION,
# ]

# TRAIN_CONFIGURATIONS = [
#     *QUICK_CONFIGURATIONS,
#     SGD_CONFIGURATION,
#     ADAM_CONFIGURATION,
# ]

TRAIN_CONFIGURATIONS = [
    SGD_CONFIGURATION,
    ADAM_CONFIGURATION,
    *GRID_CONFIGURATIONS,
]
