import flwr as fl
import sys
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
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

# Reshape for the CNN model — see model.py's docstring. Must match the same
# shape used in client.py, since the aggregated weights are shared across
# an identical architecture.
X_server = X_server.reshape(X_server.shape[0], NUM_FEATURES, 1)

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

    return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        all_shap = {}
        feature_names = None

        for client_idx, (client_proxy, fit_res) in enumerate(results):
            client_id = f"client_{client_idx + 1}"
            all_shap[client_id] = fit_res.metrics
            if feature_names is None:
                feature_names = list(fit_res.metrics.keys())

        num_examples = {
            f"client_{i+1}": fit_res.num_examples
            for i, (client_proxy, fit_res) in enumerate(results)
        }

        SW = compute_SHAP_weights(all_shap, feature_names, num_examples, shap_influence=0.3)

        if should_print(server_round):
            print(f"\n[Server] SHAP Weights (Round {server_round}):")
            for client_id, weight in SW.items():
                print(f"  {client_id}: {weight:.4f}")

        aggregated_layers = None
        for client_idx, (client_proxy, fit_res) in enumerate(results):
            client_id = f"client_{client_idx + 1}"
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
            client_id = f"client_{client_idx+1}"
            m = evaluate_res.metrics
            global_accuracy_history[client_id].append(m.get("accuracy", np.nan))
            global_precision_history[client_id].append(m.get("precision", np.nan))
            global_recall_history[client_id].append(m.get("recall", np.nan))
            global_f1_history[client_id].append(m.get("f1", np.nan))

        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if should_print(server_round):
            print(f"\n[Server] Round {server_round} — per-client LOCAL eval:")
            for client_idx, (_, evaluate_res) in enumerate(results):
                client_id = f"client_{client_idx+1}"
                m = evaluate_res.metrics
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
