# Consensus SHAP with previous-round server F1

This version keeps five separate clients and places the complete Flower
aggregation strategy inside `server.py`.

## Aggregation sequence

1. The server sends the current global parameters to all five clients.
2. Each client trains only on its local training partition.
3. Each client sends its updated parameters, class-specific SHAP vectors,
   class-availability mask, and training-example count.
4. For every client and class, the server creates an element-wise median SHAP
   reference using the other available clients. The client being scored is
   excluded from its own reference.
5. The client vector is compared with that reference using positive cosine
   similarity.
6. The previous server-validation F1 gives more importance to classes that
   the preceding global model detected poorly.
7. The similarity scores are converted to SHAP client weights and blended
   50:50 with ordinary FedAvg weights.
8. The new global model is evaluated on server validation data and by every
   client on its own local test partition.
9. The new server per-class F1 values are used in the next round.

Round 1 uses `0.05` for every server class F1, so its class importance starts
equal. There are no warm-up rounds in this version.

## Server data separation

`Server_Test_7Class.csv` is split once using a reproducible stratified 50:50
split:

- The validation half is evaluated after every aggregation. Its per-class F1
  affects the following round.
- The final-test half is evaluated only after round 10. It is also used for
  final server SHAP.

## Final post-hoc SHAP comparison

After round 10:

- The server calculates class-specific SHAP for the final global model using
  balanced samples from the untouched server final-test partition.
- Each client has already calculated its round-10 class-specific SHAP using
  balanced samples from its own local test partition.
- The server compares each client with the server for every available class.

The output includes cosine similarity, Pearson correlation, Spearman rank
correlation, mean absolute difference, top-feature match, and top-5/top-10
overlap. This is a cross-distribution comparison: differences can be caused
by both model differences and non-IID data differences.

Post-hoc SHAP runs only after final aggregation and cannot change training.

## Important files

- `server.py`: server loading, aggregation strategy, evaluation and saving.
- `class_specific_shap_weighting.py`: client SHAP calculation and consensus
  weighting mathematics.
- `posthoc_shap_analysis.py`: final server/client comparison tables.
- `client1.py` through `client5.py`: five separately runnable clients.
- `model.py`: shared seven-class neural network.
- `Labels7.py`: class and feature definitions.

`improved_shap_strategy.py` is not needed because its strategy is merged into
`server.py`.

## Running

Start the server:

```powershell
python server.py 8080
```

Then start the five clients in separate terminals:

```powershell
python client1.py 8080
python client2.py 8080
python client3.py 8080
python client4.py 8080
python client5.py 8080
```

The default output directory is:

`results_10round_7class_equal_consensus_shap_previous_server_f1`
