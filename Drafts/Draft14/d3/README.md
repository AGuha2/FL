# Eight-class SHAP-weighted federated IDS

## Files

- `create_8class_split.py` creates the five non-IID client datasets, the
  independent server dataset, and the shared scaler.
- `model.py` contains the directly reusable model definition.
- `client1.py` through `client5.py` are separate, standalone client programs.
- `server.py` performs SHAP-weighted aggregation and server-side evaluation.

The default paths match the supplied split:

- Clients: `D:\CAPSTONE\Dir_Client1.csv` ... `Dir_Client5.csv`
- Server holdout: `D:\CAPSTONE\Server_Test_Class.csv`
- Scaler: `D:\CAPSTONE\global_scaler_Class.pkl`

Run the split script once before federated training:

```powershell
python create_8class_split.py
```

Start the server:

```powershell
python server.py 8080
```

Start each client in its own terminal:

```powershell
python client1.py 8080
python client2.py 8080
python client3.py 8080
python client4.py 8080
python client5.py 8080
```

## SHAP aggregation

Each client calculates mean absolute SHAP values using a fixed sample of its
local training data. The server normalizes these vectors and forms a robust
median cross-client explanation consensus. Client SHAP weight is proportional
to non-negative cosine alignment with this consensus:

`final_weight = 0.70 * sample_size_weight + 0.30 * SHAP_alignment_weight`

This keeps the proposed SHAP novelty while preventing one highly unusual SHAP
vector from receiving excessive influence. Change `SIZE_INFLUENCE` and
`SHAP_INFLUENCE` in `server.py` only as part of a documented ablation study.

## Results

The `results` folder contains:

- server and correctly attributed client accuracy, balanced accuracy,
  macro-precision, macro-recall, and macro-F1;
- labeled server/client confusion matrices for each round;
- SHAP aggregation weights for each client and round;
- the best global checkpoint selected using server validation macro-F1;
- final server-versus-client SHAP cosine similarity, Spearman rank
  correlation, top-10 overlap, and per-feature differences.

The final SHAP comparison deliberately contrasts the final global model on the
server domain with each final local model on its client domain. This measures
explanation agreement under non-IID data, which should be stated explicitly in
the analysis.
