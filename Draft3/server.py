import flwr as fl
import sys
import numpy as np

NUM_CLIENTS = 5

def fit_config(server_round: int):
        return {"server_round": server_round}

def eval_config(server_round: int):
        return {"server_round": server_round}

class SaveModelStrategy(fl.server.strategy.FedProx):


    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        print(f"\n[Server] SHAP Summary — Round {server_round}")
        print("-" * 60)
        for client_idx, (client_proxy, fit_res) in enumerate(results):
            client_id = f"client_{client_idx + 1}"
            feats = fit_res.metrics

            top5 = sorted(feats.items(), key=lambda x: x[1], reverse=True)[:5]
            top5_str = " | ".join([f"{f}: {v:.4f}" for f, v in top5])
            print(f"  [{client_id}] top 5 → {top5_str}")
        return aggregated_weights

# Create strategy and run server
strategy = SaveModelStrategy(
    proximal_mu=1.0, 
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