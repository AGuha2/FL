import flwr as fl
import sys
import numpy as np
import json
import pandas as pd
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


NUM_CLIENTS = 2
FEATURE_NAMES = [
    'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate',
    'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'urg_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
    'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP',
    'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number'
]   # your 40 feature names in order


def cosine_similarity(a, b):
    """Cosine similarity between two 1D numpy arrays."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class EDSWeightedFedProx(fl.server.strategy.FedProx):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Log EDS and weights every round for analysis
        self.round_log = []

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        # ── Extract weights, SHAP vectors, sample counts ──────────────────
        client_weights  = []
        shap_vectors    = []
        sample_counts   = []
        client_ids      = []

        for i, (proxy, fit_res) in enumerate(results):
            # Model weights
            params = parameters_to_ndarrays(fit_res.parameters)
            client_weights.append(params)
            sample_counts.append(fit_res.num_examples)

            # SHAP vector sent as metrics — 40 values keyed shap_f0..shap_f39
            shap_vec = np.array([
                fit_res.metrics.get(f"shap_f{j}", 0.0)
                for j in range(40)
            ], dtype=np.float32)

            # Normalise to unit length
            norm = np.linalg.norm(shap_vec)
            if norm > 0:
                shap_vec = shap_vec / norm

            shap_vectors.append(shap_vec)
            client_ids.append(i + 1)

        n_total = sum(sample_counts)

        # ── Step 1: Global SHAP profile S_g ───────────────────────────────
        S_g = np.zeros(40, dtype=np.float32)
        for n_i, S_i in zip(sample_counts, shap_vectors):
            S_g += (n_i / n_total) * S_i

        # Normalise S_g
        norm_g = np.linalg.norm(S_g)
        if norm_g > 0:
            S_g = S_g / norm_g

        # ── Step 2: EDS per client ─────────────────────────────────────────
        eds_scores = []
        for S_i in shap_vectors:
            cos_sim = cosine_similarity(S_i, S_g)
            eds = 1.0 - cos_sim
            eds_scores.append(eds)

        # ── Step 3: Aggregation weights ────────────────────────────────────
        raw_weights = [
            n_i * (1.0 - eds_i)
            for n_i, eds_i in zip(sample_counts, eds_scores)
        ]
        total_raw = sum(raw_weights)

        if total_raw == 0:
            # Fallback to size-only weights if all EDS = 1
            final_weights = [n_i / n_total for n_i in sample_counts]
        else:
            final_weights = [w / total_raw for w in raw_weights]

        # ── Step 4: Weighted average of model parameters ───────────────────
        aggregated = [
            np.sum(
                [W_i * layer for W_i, layer in
                 zip(final_weights, [cw[layer_idx]
                                     for cw in client_weights])],
                axis=0
            )
            for layer_idx in range(len(client_weights[0]))
        ]

        # ── Log this round ─────────────────────────────────────────────────
        for i, cid in enumerate(client_ids):
            self.round_log.append({
                'round':        rnd,
                'client':       cid,
                'n_samples':    sample_counts[i],
                'eds':          eds_scores[i],
                'size_weight':  sample_counts[i] / n_total,
                'final_weight': final_weights[i],
            })

        print(f"\n[Round {rnd}] EDS-weighted aggregation:")
        for i, cid in enumerate(client_ids):
            print(f"  Client {cid}: "
                  f"EDS={eds_scores[i]:.4f} | "
                  f"size_w={sample_counts[i]/n_total:.3f} | "
                  f"final_w={final_weights[i]:.3f}")

        # Print top SHAP features this round
        top5_idx = np.argsort(S_g)[::-1][:5]
        print(f"  Global SHAP top 5: "
              f"{[(FEATURE_NAMES[j], round(float(S_g[j]),3)) for j in top5_idx]}")

        # Save weights
        print(f"Saving round {rnd} aggregated_weights...")
        np.savez(f"round-{rnd}-weights.npz", *aggregated)

        return ndarrays_to_parameters(aggregated), {}

    def configure_evaluate(self, rnd, parameters, client_manager):
        """Trigger full classification report on final round."""
        config = {"run_report": rnd == 30}
        eval_ins = fl.common.EvaluateIns(parameters, config)
        clients  = client_manager.sample(
            num_clients=client_manager.num_available()
        )
        return [(c, eval_ins) for c in clients]

    def aggregate_evaluate(self, rnd, results, failures):
        """Weighted average of accuracy across clients."""
        if not results:
            return None, {}
        total    = sum(r.num_examples for _, r in results)
        accuracy = sum(
            r.num_examples * r.metrics.get("accuracy", 0)
            for _, r in results
        ) / total
        print(f"[Round {rnd}] Global weighted accuracy: {accuracy:.4f}")
        return None, {"accuracy": accuracy}


# ── Save round log after training ─────────────────────────────────────────────
def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    acc   = sum(n * m.get("accuracy", 0) for n, m in metrics) / total
    return {"accuracy": acc}


strategy = EDSWeightedFedProx(
    proximal_mu=1.0,
    min_available_clients=NUM_CLIENTS,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address='localhost:' + str(sys.argv[1]),
    config=fl.server.ServerConfig(num_rounds=30),
    grpc_max_message_length=1024 * 1024 * 1024,
    strategy=strategy,
)

# ── Save round log to CSV after all rounds ────────────────────────────────────
df_log = pd.DataFrame(strategy.round_log)
df_log.to_csv("eds_round_log.csv", index=False)
print("\nRound log saved to eds_round_log.csv")
print(df_log.groupby('client')[['eds', 'final_weight']].mean().round(4))