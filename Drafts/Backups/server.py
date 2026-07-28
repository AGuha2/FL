import flwr as fl
import sys
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import numpy as np

NUM_CLIENTS = 5

def fit_config(server_round: int):
        return {"server_round": server_round}

def eval_config(server_round: int):
        return {"server_round": server_round}

def compute_SHAP_weights(all_shap, feature_names, num_examples, shap_influence=0.2):
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

    #FedAvg Weighting
    n_total = sum(num_examples[cid] for cid in client_ids)
    size_weights = np.array([num_examples[cid] / n_total for cid in client_ids])

    blended = shap_influence * weights + (1 - shap_influence) * size_weights
    blended = blended / blended.sum()

    return {client_ids[i]: float(blended[i]) for i in range(num_clients)}

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
        
        SW = compute_SHAP_weights(
        all_shap,
        feature_names,
        num_examples,
        shap_influence=0.2
    )
        
        #print(f"\n[Server] SHAP Weights (Round {server_round}):")
        #for client_id, weight in SW.items():
        #    print(f"{client_id}: {weight:.4f}")

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

            acc = evaluate_res.metrics["accuracy"]

            global_accuracy_history[client_id].append(acc)

        aggregated = super().aggregate_evaluate(
            server_round,
            results,
            failures
        )

        if server_round == 20:
            import pandas as pd

            table = pd.DataFrame(global_accuracy_history)
            table.index += 1
            table.index.name = "Round"

            print("\n")
            print("="*60)
            print("GLOBAL ACCURACY OF ALL CLIENTS")
            print("="*60)
            print(table.round(4))
            print("="*60)

            table.to_csv("global_accuracy_table.csv")

        return aggregated

# Create strategy and run server
global_accuracy_history = {
    f"client_{i+1}": []
    for i in range(NUM_CLIENTS)
}

strategy = SaveModelStrategy(
    #proximal_mu=1.0, 
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=eval_config,
)

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = 'localhost:'+str(sys.argv[1]) , 
        config=fl.server.ServerConfig(num_rounds=20) ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
)