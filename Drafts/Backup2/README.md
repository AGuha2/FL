# Seven-class FedAvg baseline

This is the comparison baseline for the seven-class proposed SHAP method.

It keeps the following identical:

- seven-class client datasets and server holdout;
- model architecture and initialization seed;
- 10 rounds and one local epoch;
- batch size, optimizer and learning-rate schedule;
- focal loss and local class weighting;
- evaluation metrics and train/test splits.

It removes:

- SHAP calculation and SHAP aggregation;
- class-specific and class-aware aggregation;
- FedProx;
- final-layer personalization.

Flower's standard FedAvg aggregates complete client models in proportion to
`num_examples`. Results are written to `results_10round_7class_fedavg`.

Use the existing `Labels7.py`, `model.py`, `Dir7_Client*.csv`,
`global_scaler_7Class.pkl`, and `Server_Test_7Class.csv`.

Run:

```powershell
python server.py 8080
```

Then run the five clients in separate terminals. Compare global server and
global client columns only with the proposed method. Do not compare this
baseline with the proposed personalized columns.
