# Reliable class-specific consensus SHAP

This experiment combines four aggregation signals:

1. FedAvg sample-size weight.
2. Per-class F1 of each submitted client model on one common balanced
   server-validation subset.
3. Per-class SHAP similarity to a leave-one-client-out median consensus.
4. Per-class F1 gain over the previous global model.

The final server test partition remains untouched until round 10.

## Round sequence

For round `r`:

1. The server sends global model `r-1` to all five clients.
2. Each client trains on only its local training partition.
3. Each client returns its updated weights, class-specific SHAP vectors,
   class-availability indicators, and number of training examples.
4. The server evaluates every submitted model on the same balanced validation
   subset, using up to 2,000 records per class.
5. For each client and class, the server compares the client SHAP vector with
   the median vector from the other clients where the class is available.
6. The previous global per-class F1 determines class difficulty.
7. The server calculates and applies the final client aggregation weights.
8. The newly aggregated global model is evaluated on server validation data
   and separately by each client on its local test partition.

Round 1 uses a previous server F1 of `0.05` for every class.

## Formula

For client `i` and class `c`:

```text
reliability(i,c) = F1 of submitted client model on server validation class c
similarity(i,c)  = cosine(client SHAP, median SHAP from other clients)
gain(i,c)        = max(0, reliability(i,c) - previous global F1(c))

class score(i,c) =
    0.50 * reliability(i,c)
  + 0.30 * similarity(i,c)
  + 0.20 * gain(i,c)
```

Poorly detected classes receive greater importance:

```text
importance(c) proportional to 1 / max(previous global F1(c), 0.05)
```

The resulting client score is converted to a softmax SHAP/utility weight and
combined with FedAvg:

```text
final weight = 0.50 * FedAvg weight + 0.50 * SHAP/utility weight
```

Weights remain bounded to `[0.12, 0.28]` and smoothed between rounds.

## Privacy and data separation

Clients do not send raw records or exact class counts. The server already
receives the submitted model parameters as part of federated learning and
evaluates those parameters locally on server validation data.

The server CSV is split reproducibly:

- 50% validation: per-round evaluation and client-update reliability.
- 50% final test: final round-10 metrics and final server SHAP only.

This is federated data locality, not formal differential privacy.

## Final post-hoc SHAP

After round 10, the final global model's SHAP values on the server final-test
partition are compared with each client's round-10 SHAP values calculated on
that client's local test partition.

This is a cross-distribution comparison. Differences can reflect both model
differences and non-IID data differences.

## Outputs

The server saves to:

`results_10round_7class_equal_reliable_consensus_shap`

`aggregation_weights.csv` includes, for every client and class:

- `importance::<class>`
- `previous_server_f1::<class>`
- `client_validation_f1::<class>`
- `similarity::<class>`
- `gain::<class>`
- final client weight

Clients save final local artifacts under:

`client_posthoc_results/Reliable_Consensus_SHAP/client_<id>/`

## Running

```powershell
python server.py 8080
```

Start the clients in five separate terminals:

```powershell
python client1.py 8080
python client2.py 8080
python client3.py 8080
python client4.py 8080
python client5.py 8080
```
