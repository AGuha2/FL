import flwr as fl
import sys
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras # type: ignore
import joblib
import shap # type: ignore
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from Labels import le, NUM_CLASSES, NUM_FEATURES
from model import build_model

NUM_CLIENTS = 5
NUM_ROUNDS = 30          # was 20 — more rounds since local epochs also increased
PROXIMAL_MU = 0.1    # FedProx penalty strength; 0.0 disables it entirely

SERVER_TEST_PATH = r'D:\CAPSTONE\Server_Test.csv'
SCALER_PATH = r'D:\CAPSTONE\global_scaler.pkl'

# Only print full metric breakdowns on these rounds, to avoid flooding the
# console every single round. Full history is still saved to CSV either way.
PRINT_EVERY = 5


def should_print(server_round: int) -> bool:
    return server_round == 1 or server_round == NUM_ROUNDS or server_round % PRINT_EVERY == 0


def compute_all_metrics(y_true, y_pred):
    """accuracy + macro-averaged precision/recall/F1. Macro (not weighted) is
    used deliberately: with 34 heavily-imbalanced, Dirichlet-skewed classes,
    a weighted average would be dominated by majority classes and hide poor
    performance on the rare classes concentrated on individual clients."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return acc, precision, recall, f1


def fit_config(server_round: int):
    # proximal_mu is read by the client's custom training loop to add the
    # FedProx penalty term. This is NOT the same as passing proximal_mu to
    # fl.server.strategy.FedProx — that built-in class only forwards this
    # value into the config; it never applies the penalty itself. The actual
    # penalty has to be computed client-side, which is why we're threading it
    # through here manually alongside our custom SHAP-weighted aggregation.
    return {"server_round": server_round, "proximal_mu": PROXIMAL_MU}


def eval_config(server_round: int):
    return {"server_round": server_round}


def compute_SHAP_weights(all_shap, feature_names, num_examples, shap_influence=0.3):
    client_ids = list(all_shap.keys())
    num_clients = len(client_ids)

    shap_matrix = np.array([
        [all_shap[cid].get(f, 0.0) for f in feature_names]
        for cid in client_ids
    ])

    norms = np.linalg.norm(shap_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    shap_normed = shap_matrix / norms

    sim_matrix = shap_normed @ shap_normed.T

    S_w = np.zeros(num_clients)
    for i in range(num_clients):
        other_sims = [sim_matrix[i][j] for j in range(num_clients) if j != i]
        S_w[i] = 1.0 - np.mean(other_sims)

    total = S_w.sum()
    weights = S_w / total if total > 0 else np.ones(num_clients) / num_clients

    n_total = sum(num_examples[cid] for cid in client_ids)
    size_weights = np.array([num_examples[cid] / n_total for cid in client_ids])

    blended = shap_influence * weights + (1 - shap_influence) * size_weights
    blended = blended / blended.sum()

    return {client_ids[i]: float(blended[i]) for i in range(num_clients)}


# ── Centralized server-side evaluation on the held-out Server_Test.csv ────────
# This is the distribution-neutral, all-classes-represented evaluation set —
# unlike each client's local evaluate(), which only ever sees that client's
# own Dirichlet-skewed slice. This is the trustworthy number for tracking
# true global generalization across training.
print("[Server] Loading centralized server-side test set...")
server_df = pd.read_csv(SERVER_TEST_PATH)
server_df.replace([np.inf, -np.inf], np.nan, inplace=True)
server_df = server_df.dropna()

X_server = server_df.drop(columns=['Label'])
y_server = server_df['Label'].str.strip().str.upper()
y_server = le.transform(y_server)

scaler = joblib.load(SCALER_PATH)
X_server = scaler.transform(X_server)
# X_server stays flat 2D here — any model-specific reshaping happens inside
# model.py itself, so swapping architectures never requires touching this file.

print(f"[Server] Server test set loaded: {X_server.shape[0]:,} rows, "
      f"{len(np.unique(y_server))} classes")

# Model architecture must exactly match the client's model — aggregated
# weights (which include BatchNorm's non-trainable moving stats, since
# clients return full model.get_weights()) have to load into an identical
# layer structure or set_weights() will raise a shape/order mismatch.
eval_model = build_model(input_features=NUM_FEATURES, num_classes=NUM_CLASSES,
                          output_activation='softmax')
eval_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

server_metrics_history = []

# ── Final-round comparative SHAP: each client vs. the aggregated server model ──
# Reuses each client's own last-round SHAP output (already computed in their
# fit() for aggregation weighting — no extra client-side work needed) and
# additionally runs SHAP once on the server's own final aggregated model
# against Server_Test.csv, so we can see how each client's local
# feature-importance profile compares to what the merged global model
# actually ended up weighting. Only computed once, at the last round — not
# a per-round cost.
# NOTE: like every SHAP call in this project, DeepExplainer requires a
# feed-forward architecture (no LSTM/recurrent layers) — see model.py.
_final_shap_rng = np.random.RandomState(42)
FINAL_SHAP_BG_SIZE = min(100, X_server.shape[0])
FINAL_SHAP_SAMPLE_SIZE = min(500, X_server.shape[0])
_bg_idx = _final_shap_rng.choice(X_server.shape[0], FINAL_SHAP_BG_SIZE, replace=False)
_sample_idx = _final_shap_rng.choice(X_server.shape[0], FINAL_SHAP_SAMPLE_SIZE, replace=False)
final_shap_background = X_server[_bg_idx]
final_shap_sample = X_server[_sample_idx]
final_shap_feature_names = server_df.drop(columns=['Label']).columns.tolist()

final_client_shap = {}   # populated at the last round, inside aggregate_fit
final_server_shap = {}   # populated at the last round, inside evaluate_fn


def evaluate_fn(server_round, parameters, config):
    """Centralized evaluation hook, called by the strategy on the aggregated
    global model each round — independent of any single client's data."""
    eval_model.set_weights(parameters)

    loss, _ = eval_model.evaluate(X_server, y_server, verbose=0)
    y_pred = np.argmax(eval_model.predict(X_server, verbose=0), axis=1)
    accuracy, precision, recall, f1 = compute_all_metrics(y_server, y_pred)

    server_metrics_history.append({
        "round": server_round,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    if should_print(server_round):
        print(f"\n[Server] === CENTRALIZED TEST SET (Round {server_round}) ===")
        print(f"  loss: {loss:.4f} | accuracy: {accuracy:.4f} | "
              f"precision(macro): {precision:.4f} | recall(macro): {recall:.4f} | "
              f"f1(macro): {f1:.4f}")

    if server_round == NUM_ROUNDS:
        print("\n[Server] Computing SHAP on the final aggregated global model...")
        explainer = shap.DeepExplainer(eval_model, final_shap_background)
        shap_values = explainer.shap_values(final_shap_sample, check_additivity=False)
        shap_array = np.array(shap_values)                        # (classes, samples, features[, channels])
        mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 1))  # (features,) or (features, channels)
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.mean(axis=-1)            # collapse any extra channel dim
        num_feats = mean_abs_shap.shape[0]
        final_server_shap.update({
            feat: float(mean_abs_shap[i])
            for i, feat in enumerate(final_shap_feature_names[:num_feats])
        })

    return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        all_shap = {}
        feature_names = None
        client_ids_by_idx = {}  # result index -> real client_id, for reuse below

        for client_idx, (client_proxy, fit_res) in enumerate(results):
            # Prefer the client's self-reported real CLIENT_ID (stable across
            # rounds) over response order (NOT stable — Flower's results list
            # order depends on which client answers first that round, so
            # "client_1" could be a different physical client every round).
            metrics = dict(fit_res.metrics)
            real_id = metrics.pop("__client_id__", None)
            client_id = f"client_{int(real_id)}" if real_id is not None else f"client_{client_idx + 1}"
            client_ids_by_idx[client_idx] = client_id

            all_shap[client_id] = metrics
            if feature_names is None:
                feature_names = list(metrics.keys())

        num_examples = {
            client_ids_by_idx[i]: fit_res.num_examples
            for i, (client_proxy, fit_res) in enumerate(results)
        }

        SW = compute_SHAP_weights(all_shap, feature_names, num_examples, shap_influence=0.3)

        if server_round == NUM_ROUNDS:
            final_client_shap.update(all_shap)

        if should_print(server_round):
            print(f"\n[Server] SHAP Weights (Round {server_round}):")
            for client_id, weight in SW.items():
                print(f"  {client_id}: {weight:.4f}")

        aggregated_layers = None
        for client_idx, (client_proxy, fit_res) in enumerate(results):
            client_id = client_ids_by_idx[client_idx]
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            w = SW[client_id]

            if aggregated_layers is None:
                aggregated_layers = [layer * w for layer in client_weights]
            else:
                for i, layer in enumerate(client_weights):
                    aggregated_layers[i] += layer * w

        aggregated_parameters = ndarrays_to_parameters(aggregated_layers)
        return aggregated_parameters, {}

    def aggregate_evaluate(self, server_round, results, failures):
        for client_idx, (_, evaluate_res) in enumerate(results):
            m = evaluate_res.metrics
            real_id = m.get("client_id")
            client_id = f"client_{int(real_id)}" if real_id is not None else f"client_{client_idx + 1}"
            global_accuracy_history[client_id].append(m.get("accuracy", np.nan))
            global_precision_history[client_id].append(m.get("precision", np.nan))
            global_recall_history[client_id].append(m.get("recall", np.nan))
            global_f1_history[client_id].append(m.get("f1", np.nan))

        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if should_print(server_round):
            print(f"\n[Server] Round {server_round} — per-client LOCAL eval:")
            for client_idx, (_, evaluate_res) in enumerate(results):
                m = evaluate_res.metrics
                real_id = m.get("client_id")
                client_id = f"client_{int(real_id)}" if real_id is not None else f"client_{client_idx + 1}"
                print(f"  {client_id}: accuracy={m.get('accuracy', float('nan')):.4f} | "
                      f"precision={m.get('precision', float('nan')):.4f} | "
                      f"recall={m.get('recall', float('nan')):.4f} | "
                      f"f1={m.get('f1', float('nan')):.4f}")

        if server_round == NUM_ROUNDS:
            acc_table = pd.DataFrame(global_accuracy_history)
            prec_table = pd.DataFrame(global_precision_history)
            rec_table = pd.DataFrame(global_recall_history)
            f1_table = pd.DataFrame(global_f1_history)
            for t in (acc_table, prec_table, rec_table, f1_table):
                t.index += 1
                t.index.name = "Round"

            print("\n")
            print("=" * 60)
            print("GLOBAL ACCURACY OF ALL CLIENTS (local eval)")
            print("=" * 60)
            print(acc_table.round(4))

            print("\n" + "=" * 60)
            print("GLOBAL PRECISION (macro) OF ALL CLIENTS (local eval)")
            print("=" * 60)
            print(prec_table.round(4))

            print("\n" + "=" * 60)
            print("GLOBAL RECALL (macro) OF ALL CLIENTS (local eval)")
            print("=" * 60)
            print(rec_table.round(4))

            print("\n" + "=" * 60)
            print("GLOBAL F1 (macro) OF ALL CLIENTS (local eval)")
            print("=" * 60)
            print(f1_table.round(4))
            print("=" * 60)

            acc_table.to_csv("global_accuracy_table.csv")
            prec_table.to_csv("global_precision_table.csv")
            rec_table.to_csv("global_recall_table.csv")
            f1_table.to_csv("global_f1_table.csv")

            # Centralized, distribution-neutral results on Server_Test.csv —
            # the number that actually reflects global generalization.
            server_table = pd.DataFrame(server_metrics_history).set_index("round")
            server_table.index.name = "Round"
            print("\n" + "=" * 60)
            print("CENTRALIZED SERVER TEST SET METRICS (all rounds)")
            print("=" * 60)
            print(server_table.round(4))
            print("=" * 60)
            server_table.to_csv("server_test_metrics.csv")

        return aggregated


global_accuracy_history = {f"client_{i+1}": [] for i in range(NUM_CLIENTS)}
global_precision_history = {f"client_{i+1}": [] for i in range(NUM_CLIENTS)}
global_recall_history = {f"client_{i+1}": [] for i in range(NUM_CLIENTS)}
global_f1_history = {f"client_{i+1}": [] for i in range(NUM_CLIENTS)}

strategy = SaveModelStrategy(
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=eval_config,
    evaluate_fn=evaluate_fn,
)

fl.server.start_server(
    server_address='localhost:' + str(sys.argv[1]),
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    grpc_max_message_length=1024 * 1024 * 1024,
    strategy=strategy
)

# ── Final-round SHAP comparison: each client's local model vs. the server's
#    resulting aggregated global model ──────────────────────────────────────
if final_client_shap and final_server_shap:
    all_cols = dict(final_client_shap)
    all_cols['server'] = final_server_shap

    comparison_df = pd.DataFrame(all_cols).fillna(0.0)
    comparison_df.index.name = 'feature'
    comparison_df = comparison_df.sort_values('server', ascending=False)

    print("\n" + "=" * 60)
    print("FINAL-ROUND SHAP COMPARISON: clients vs. aggregated server model")
    print("(top 20 features, ranked by server importance)")
    print("=" * 60)
    print(comparison_df.round(4).head(20))
    comparison_df.to_csv("final_shap_comparison.csv")

    # Quantify how aligned each client's feature-importance profile is with
    # the server's final model, via cosine similarity of their SHAP vectors —
    # a client far out of line with the server here is a client whose local
    # model learned to rely on quite different features than what the
    # aggregated global model ended up weighting.
    server_vec = comparison_df['server'].values
    server_norm = np.linalg.norm(server_vec)

    print("\nCosine similarity of each client's final SHAP profile to the server's:")
    similarity_rows = []
    for client_id in final_client_shap.keys():
        client_vec = comparison_df[client_id].values
        denom = np.linalg.norm(client_vec) * server_norm
        sim = float(np.dot(client_vec, server_vec) / denom) if denom > 0 else 0.0
        similarity_rows.append({"client": client_id, "cosine_similarity_to_server": sim})
        print(f"  {client_id}: {sim:.4f}")

    pd.DataFrame(similarity_rows).to_csv("final_shap_client_server_similarity.csv", index=False)
    print("=" * 60)
    print("Saved: final_shap_comparison.csv, final_shap_client_server_similarity.csv")
