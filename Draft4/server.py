import flwr as fl
import sys
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import numpy as np

NUM_CLIENTS = 5

def fit_config(server_round: int):
        return {"server_round": server_round}

def eval_config(server_round: int):
        return {"server_round": server_round}

def compute_SHAP_weights(all_shap, feature_names):
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
    S_w = np.zeros(num_clients)
    for i in range(num_clients):
        other_sims = [sim_matrix[i][j] for j in range(num_clients) if j != i]
        S_w[i] = 1.0 - np.mean(other_sims)

    # normalise to sum to 1
    total = S_w.sum()
    weights = S_w / total if total > 0 else np.ones(num_clients) / num_clients

    return {client_ids[i]: float(weights[i]) for i in range(num_clients)}

class SaveModelStrategy(fl.server.strategy.FedProx):


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
        
        SW = compute_SHAP_weights(all_shap, feature_names)

        print(f"\n[Server] SHAP Weights (Round {server_round}):")
        for client_id, weight in SW.items():
            print(f"{client_id}: {weight:.4f}")

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

# Create strategy and run server
strategy = SaveModelStrategy(
    proximal_mu=1.0, 
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=eval_config,
)

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = 'localhost:'+str(sys.argv[1]) , 
        config=fl.server.ServerConfig(num_rounds=10) ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
)