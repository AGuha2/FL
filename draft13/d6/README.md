# Controlled FL + SHAP experiment suite

Copy these files into a new folder beside your existing Draft12 code. Keep your existing `Labels.py` and the same data files under `D:\CAPSTONE`.

## Experiments

| ID | Purpose | Class weights | FedProx | Aggregation |
|---|---|---:|---:|---|
| E0 | Clean baseline | No | No | FedAvg by data size |
| E1 | Isolate class weighting | Yes | No | FedAvg |
| E2 | Isolate client-drift control | Yes | μ=0.01 | FedAvg |
| E3 | Test SHAP alone | Yes | No | 85% size + 15% SHAP consensus |
| E4 | SHAP plus FedProx | Yes | μ=0.01 | 85% size + 15% SHAP consensus |
| E5 | Proposed method | Yes | μ=0.01 | 70% size + 15% common-validation F1 + 15% SHAP consensus |

All short experiments use 10 rounds and one local epoch. SHAP uses 20 background rows and 50 explanation rows, and is recalculated only in rounds 1, 5, and 10.

## Run in PowerShell

Open six terminals and activate your environment in each.

```powershell
$env:EXPERIMENT="E0"
python server.py 8080
```

In each client terminal, set the same experiment first:

```powershell
$env:EXPERIMENT="E0"
python client1.py 8080
```

Repeat for `client2.py` through `client5.py`.

For the next experiment, close all six processes, change `E0` to `E1`, and restart everything.

## Recommended order

1. E0
2. E1
3. E2
4. E3
5. E4
6. E5

Do not skip the baseline experiments. They establish whether class weighting, FedProx, and SHAP each add value.

## Choose the winner

Use `results_E*/server_metrics.csv`. Compare the best validation macro F1 and then report the corresponding test metrics. Prioritise:

1. Test macro F1
2. Test balanced accuracy
3. Test macro recall
4. Test accuracy

Do not select a method only because it has the highest ordinary accuracy.

## Model comparison

The default is an MLP with Layer Normalisation because the inputs are tabular. After identifying the best aggregation experiment, change `model_type` in `experiment_config.py` from `mlp` to `cnn` and rerun only the best experiment. This avoids doubling every experiment.
