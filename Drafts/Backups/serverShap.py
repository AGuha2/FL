import flwr as fl
import sys
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

def fit_config(server_round: int):
    return {"server_round": server_round}

def eval_config(server_round: int):
    return {"server_round": server_round}


def compute_uniqueness_weights(all_shap, feature_names):
    client_ids = list(all_shap.keys())
    num_clients = len(client_ids)

    # build (num_clients, num_features) matrix
    shap_matrix = np.array([
        [all_shap[cid].get(f, 0.0) for f in feature_names]
        for cid in client_ids
    ])

    # normalise each row to unit vector
    norms = np.linalg.norm(shap_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    shap_normed = shap_matrix / norms

    # cosine similarity matrix (num_clients, num_clients)
    sim_matrix = shap_normed @ shap_normed.T

    # uniqueness = 1 - mean cosine similarity to all other clients
    uniqueness = np.zeros(num_clients)
    for i in range(num_clients):
        other_sims = [sim_matrix[i][j] for j in range(num_clients) if j != i]
        uniqueness[i] = 1.0 - np.mean(other_sims)

    # normalise to sum to 1
    total = uniqueness.sum()
    weights = uniqueness / total if total > 0 else np.ones(num_clients) / num_clients

    return {client_ids[i]: float(weights[i]) for i in range(num_clients)}


class SHAPFedProxStrategy(fl.server.strategy.FedProx):

    def aggregate_fit(self, server_round, results, failures):

        if not results:
            return None, {}

        # ── 1. collect SHAP metrics from each client ──────────────
        all_shap = {}
        feature_names = None

        for client_idx, (client_proxy, fit_res) in enumerate(results):
            client_id = f"client_{client_idx + 1}"
            all_shap[client_id] = fit_res.metrics
            if feature_names is None:
                feature_names = list(fit_res.metrics.keys())

        # ── 2. compute uniqueness weights from SHAP ───────────────
        uniqueness_weights = compute_uniqueness_weights(all_shap, feature_names)

        print(f"\n[Server] SHAP uniqueness weights — Round {server_round}")
        print("-" * 55)
        for client_id, w in uniqueness_weights.items():
            shap = all_shap[client_id]
            top_feat = max(shap, key=shap.get)
            top_val = shap[top_feat]
            print(f"  [{client_id}] weight={w:.4f} | "
                  f"top feature={top_feat} ({top_val:.4f})")

        # ── 3. aggregate using SHAP uniqueness weights ────────────
        aggregated_layers = None
        for client_idx, (client_proxy, fit_res) in enumerate(results):
            client_id = f"client_{client_idx + 1}"
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            w = uniqueness_weights[client_id]

            if aggregated_layers is None:
                aggregated_layers = [layer * w for layer in client_weights]
            else:
                for i, layer in enumerate(client_weights):
                    aggregated_layers[i] += layer * w

        aggregated_parameters = ndarrays_to_parameters(aggregated_layers)

        print(f"  Aggregation complete — round {server_round}")

        return aggregated_parameters, {}


strategy = SHAPFedProxStrategy(
    proximal_mu=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=eval_config,
)

fl.server.start_server(
    server_address='localhost:' + str(sys.argv[1]),
    config=fl.server.ServerConfig(num_rounds=10),
    grpc_max_message_length=1024 * 1024 * 1024,
    strategy=strategy
)