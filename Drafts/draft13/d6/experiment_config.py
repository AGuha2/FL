import os


EXPERIMENTS = {
    "E0": {
        "experiment": "E0",
        "aggregation": "fedavg",
        "num_rounds": 10,
        "local_epochs": 1,
        "use_class_weights": False,
        "max_class_weight": 5.0,
        "min_samples_for_weighting": 20,
        "proximal_mu": 0.0,
        "initial_lr": 0.0001,
        "lr_decay": 0.95,
        "batch_size": 256,
        "model_type": "cnn",
        "shap_rounds": {1, 5, 10},
        "shap_background_size": 20,
        "shap_sample_size": 50,
        "validation_sample_size": 5000,
        "shap_reference_size": 1000,
    },

    "E1": {
        "experiment": "E1",
        "aggregation": "fedavg",
        "num_rounds": 10,
        "local_epochs": 1,
        "use_class_weights": True,
        "max_class_weight": 5.0,
        "min_samples_for_weighting": 20,
        "proximal_mu": 0.0,
        "initial_lr": 0.0001,
        "lr_decay": 0.95,
        "batch_size": 256,
        "model_type": "cnn",
        "shap_rounds": {1, 5, 10},
        "shap_background_size": 20,
        "shap_sample_size": 50,
        "validation_sample_size": 5000,
        "shap_reference_size": 1000,
    },

    "E2": {
        "experiment": "E2",
        "aggregation": "fedavg",
        "num_rounds": 10,
        "local_epochs": 1,
        "use_class_weights": True,
        "max_class_weight": 5.0,
        "min_samples_for_weighting": 20,
        "proximal_mu": 0.01,
        "initial_lr": 0.0001,
        "lr_decay": 0.95,
        "batch_size": 256,
        "model_type": "cnn",
        "shap_rounds": {1, 5, 10},
        "shap_background_size": 20,
        "shap_sample_size": 50,
        "validation_sample_size": 5000,
        "shap_reference_size": 1000,
    },

    "E3": {
        "experiment": "E3",
        "aggregation": "shap_consensus",
        "num_rounds": 10,
        "local_epochs": 1,
        "use_class_weights": True,
        "max_class_weight": 5.0,
        "min_samples_for_weighting": 20,
        "proximal_mu": 0.0,
        "initial_lr": 0.0001,
        "lr_decay": 0.95,
        "batch_size": 256,
        "model_type": "cnn",
        "shap_rounds": {1, 5, 10},
        "shap_background_size": 20,
        "shap_sample_size": 50,
        "validation_sample_size": 5000,
        "shap_reference_size": 1000,
        "shap_influence": 0.20,
    },

    "E4": {
        "experiment": "E4",
        "aggregation": "shap_consensus",
        "num_rounds": 10,
        "local_epochs": 1,
        "use_class_weights": True,
        "max_class_weight": 5.0,
        "min_samples_for_weighting": 20,
        "proximal_mu": 0.01,
        "initial_lr": 0.0001,
        "lr_decay": 0.95,
        "batch_size": 256,
        "model_type": "cnn",
        "shap_rounds": {1, 5, 10},
        "shap_background_size": 20,
        "shap_sample_size": 50,
        "validation_sample_size": 5000,
        "shap_reference_size": 1000,
        "shap_influence": 0.20,
    },

    "E5": {
        "experiment": "E5",
        "aggregation": "hybrid",
        "num_rounds": 10,
        "local_epochs": 1,
        "use_class_weights": True,
        "max_class_weight": 5.0,
        "min_samples_for_weighting": 20,
        "proximal_mu": 0.01,
        "initial_lr": 0.0001,
        "lr_decay": 0.95,
        "batch_size": 256,
        "model_type": "cnn",
        "shap_rounds": {1, 5, 10},
        "shap_background_size": 20,
        "shap_sample_size": 50,
        "validation_sample_size": 5000,
        "shap_reference_size": 1000,
        "shap_influence": 0.15,
        "performance_influence": 0.15,
    },

    # New proposed method
    "E6": {
        "experiment": "E6",
        "aggregation": "performance_shap_utility",
        "num_rounds": 10,
        "local_epochs": 1,
        "use_class_weights": True,
        "max_class_weight": 5.0,
        "min_samples_for_weighting": 20,
        "proximal_mu": 0.01,
        "initial_lr": 0.0001,
        "lr_decay": 0.95,
        "batch_size": 256,
        "model_type": "cnn",

        # SHAP is recalculated only in these rounds.
        "shap_rounds": {1, 5, 10},

        # All clients use the same server-derived reference set.
        "shap_background_size": 20,
        "shap_sample_size": 50,
        "shap_reference_size": 1000,

        "validation_sample_size": 5000,

        # Final weight:
        # 80% data-size weight + 20% SHAP-performance utility.
        "size_influence": 0.80,
        "utility_influence": 0.20,

        # SHAP quality:
        # 70% consensus agreement + 30% temporal stability.
        "consensus_influence": 0.70,
        "stability_influence": 0.30,
    },

# ============================================================
# E6A — Increase SHAP-performance utility influence
# ============================================================

"E6A": {
    "experiment": "E6A",
    "aggregation": "performance_shap_utility",

    "num_rounds": 10,
    "local_epochs": 1,

    "use_class_weights": True,
    "max_class_weight": 5.0,
    "min_samples_for_weighting": 20,

    "proximal_mu": 0.01,

    "initial_lr": 0.0001,
    "lr_decay": 0.95,
    "batch_size": 256,

    "model_type": "cnn",

    "shap_rounds": {1, 5, 10},
    "shap_background_size": 20,
    "shap_sample_size": 50,
    "shap_reference_size": 1000,

    "validation_sample_size": 5000,

    # Changed from 80/20 to 70/30
    "size_influence": 0.70,
    "utility_influence": 0.30,

    "consensus_influence": 0.70,
    "stability_influence": 0.30,
},


# ============================================================
# E6B — Two local training epochs
# ============================================================

"E6B": {
    "experiment": "E6B",
    "aggregation": "performance_shap_utility",

    "num_rounds": 10,

    # Changed from 1 to 2
    "local_epochs": 2,

    "use_class_weights": True,
    "max_class_weight": 5.0,
    "min_samples_for_weighting": 20,

    "proximal_mu": 0.01,

    "initial_lr": 0.0001,
    "lr_decay": 0.95,
    "batch_size": 256,

    "model_type": "cnn",

    "shap_rounds": {1, 5, 10},
    "shap_background_size": 20,
    "shap_sample_size": 50,
    "shap_reference_size": 1000,

    "validation_sample_size": 5000,

    "size_influence": 0.80,
    "utility_influence": 0.20,

    "consensus_influence": 0.70,
    "stability_influence": 0.30,
},


# ============================================================
# E6C — Extend training to 20 rounds
# ============================================================

"E6C": {
    "experiment": "E6C",
    "aggregation": "performance_shap_utility",

    # Changed from 10 to 20
    "num_rounds": 20,
    "local_epochs": 1,

    "use_class_weights": True,
    "max_class_weight": 5.0,
    "min_samples_for_weighting": 20,

    "proximal_mu": 0.01,

    "initial_lr": 0.0001,
    "lr_decay": 0.95,
    "batch_size": 256,

    "model_type": "cnn",

    # SHAP calculated only four times
    "shap_rounds": {1, 5, 10, 15, 20},
    "shap_background_size": 20,
    "shap_sample_size": 50,
    "shap_reference_size": 1000,

    "validation_sample_size": 5000,

    "size_influence": 0.80,
    "utility_influence": 0.20,

    "consensus_influence": 0.70,
    "stability_influence": 0.30,
},
}

EXPERIMENT_NAME = os.environ.get(
    "EXPERIMENT",
    "E6",
).upper()


if EXPERIMENT_NAME not in EXPERIMENTS:
    raise ValueError(
        f"Unknown experiment '{EXPERIMENT_NAME}'. "
        f"Valid experiments: {sorted(EXPERIMENTS)}"
    )


CFG = EXPERIMENTS[EXPERIMENT_NAME]

