import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from Labels import le, NUM_CLASSES, NUM_FEATURES
from model import build_model
import shap

from sklearn.utils.class_weight import compute_class_weight
import joblib

CLIENT_ID = 1

# Only print full metric breakdowns on these rounds, to avoid flooding the
# console every single round. Server has its own independent PRINT_EVERY.
PRINT_EVERY = 5

def should_print(round_num):
    return round_num == 1 or round_num % PRINT_EVERY == 0

# Load dataset
df = pd.read_csv(rf'D:\CAPSTONE\Dir_Client{CLIENT_ID}.csv')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

print(df.head())

X = df.drop(columns=['Label'])
y = df['Label'].str.strip().str.upper()
y = le.transform(y)

scaler = joblib.load(r'D:\CAPSTONE\global_scaler.pkl')
X = scaler.transform(X)

# ── Reshape into a sequence for the CNN-LSTM model ─────────────────────────────
# build_model's TimeDistributed/LSTM stack expects (samples, timesteps,
# features, 1). We don't have genuine multi-timestep windowed data yet, so
# each row is reshaped as a single-timestep sequence — this makes the
# architecture runnable, but the LSTM layers get no real temporal signal
# until timesteps > 1 with actual sequential data.
NUM_TIMESTEPS = 1
X = X.reshape(X.shape[0], NUM_TIMESTEPS, NUM_FEATURES, 1)

try:
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    print(f"[Client] Warning: some classes have < 2 samples, "
          f"falling back to non-stratified split")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"Train: {len(x_train)} | Test: {len(x_test)}")

# ── Class weighting, capped and zeroed for extremely rare classes ─────────────
present_classes = np.unique(y_train)
cw = compute_class_weight(class_weight='balanced', classes=present_classes, y=y_train)

MAX_WEIGHT = 8.0
cw = np.clip(cw, a_min=None, a_max=MAX_WEIGHT)

# For classes with too few samples to learn reliably, don't apply any weighting
# boost at all — a class with 1-5 samples can dominate a batch's loss whenever
# it appears, causing large, unstable local loss spikes.
class_counts = pd.Series(y_train).value_counts()
MIN_SAMPLES_FOR_WEIGHTING = 10
for i, c in enumerate(present_classes):
    if class_counts.get(c, 0) < MIN_SAMPLES_FOR_WEIGHTING:
        cw[i] = 1.0

class_weight_dict = {int(c): float(w) for c, w in zip(present_classes, cw)}
print(f"[Client {CLIENT_ID}] class_weight_dict: {class_weight_dict}")

model = build_model(input_features=NUM_FEATURES, num_classes=NUM_CLASSES,
                     num_timesteps=NUM_TIMESTEPS, output_activation='softmax')
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ── SHAP setup: background + a FIXED, stratified sample ────────────────────────
# A plain random draw can miss rare classes entirely (e.g. a class with only
# 5 samples out of 240,000 has roughly a 1-in-500 chance of appearing at all
# in a random 500-row draw). This builds a sample that mirrors the client's
# real class distribution while guaranteeing every present class appears at
# least MIN_PER_CLASS times, so rare-class feature importance isn't silently
# dropped from the SHAP calculation.
def build_stratified_shap_sample(x_train, y_train, sample_size, min_per_class=2, seed=42):
    rng = np.random.RandomState(seed)
    present_classes, counts = np.unique(y_train, return_counts=True)
    proportions = counts / counts.sum()

    # proportional allocation, floored at min_per_class (but never more than
    # the class actually has)
    alloc = np.maximum(
        np.round(proportions * sample_size).astype(int),
        np.minimum(min_per_class, counts)
    )

    # if the floor pushed the total over budget, trim the largest allocations
    # down (never below their min) until we're back at sample_size
    while alloc.sum() > sample_size:
        i = np.argmax(alloc - np.minimum(min_per_class, counts))
        alloc[i] -= 1

    idx_parts = []
    for cls, n in zip(present_classes, alloc):
        cls_idx = np.where(y_train == cls)[0]
        take = min(n, len(cls_idx))
        chosen = rng.choice(cls_idx, size=take, replace=False)
        idx_parts.append(chosen)

    sample_idx = np.concatenate(idx_parts)
    rng.shuffle(sample_idx)
    return x_train[sample_idx]


bg_size = min(50, x_train.shape[0])
bg_idx = np.random.choice(x_train.shape[0], bg_size, replace=False)
background = x_train[bg_idx]

fixed_sample_size = min(150, x_train.shape[0])
x_sample = build_stratified_shap_sample(x_train, y_train, fixed_sample_size, min_per_class=2)
print(f"[Client {CLIENT_ID}] SHAP sample: {x_sample.shape[0]} rows covering "
      f"{len(np.unique(y_train))} classes present in local data")

# ── Local-epoch / learning-rate schedule ───────────────────────────────────────
LOCAL_EPOCHS = 2          # was 1 — more local training per round
INITIAL_LR = 1e-4
LR_DECAY_RATE = 0.95      # LR shrinks each round so updates don't stay too large
                          # once the global model is already converging

optimizer = keras.optimizers.Adam(learning_rate=INITIAL_LR, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

loss_fn = keras.losses.SparseCategoricalCrossentropy()


def make_batches(x, y, batch_size, class_weight_dict):
    n = x.shape[0]
    idx = np.random.permutation(n)
    sample_weights = np.array([class_weight_dict.get(int(label), 1.0) for label in y])
    for start in range(0, n, batch_size):
        b_idx = idx[start:start + batch_size]
        yield x[b_idx], y[b_idx], sample_weights[b_idx]


def compute_all_metrics(y_true, y_pred):
    """accuracy + macro-averaged precision/recall/F1. Macro (not weighted) is
    used deliberately: with 34 heavily-imbalanced classes, a weighted average
    would be dominated by majority classes and hide poor performance on the
    rare, Dirichlet-concentrated classes we care about."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return acc, precision, recall, f1


def train_one_epoch_fedprox(model, x_train, y_train, class_weight_dict,
                             global_weights, proximal_mu, batch_size=256):
    """Custom training loop that adds the FedProx proximal penalty:
        loss = CE_loss + (proximal_mu / 2) * sum((w_local - w_global)^2)
    global_weights are the weights received from the server at the START of
    this round (before any local training) — the anchor point local weights
    are penalized for drifting away from.
    """
    global_weights_tf = [tf.constant(w) for w in global_weights]

    for xb, yb, swb in make_batches(x_train, y_train, batch_size, class_weight_dict):
        xb = tf.convert_to_tensor(xb, dtype=tf.float32)
        yb = tf.convert_to_tensor(yb, dtype=tf.int32)
        swb = tf.convert_to_tensor(swb, dtype=tf.float32)

        with tf.GradientTape() as tape:
            preds = model(xb, training=True)
            per_sample_loss = loss_fn(yb, preds)
            ce_loss = tf.reduce_mean(per_sample_loss * swb)

            if proximal_mu > 0:
                prox_term = tf.add_n([
                    tf.reduce_sum(tf.square(w_local - w_global))
                    for w_local, w_global in zip(model.trainable_variables, global_weights_tf)
                ])
                loss = ce_loss + (proximal_mu / 2.0) * prox_term
            else:
                loss = ce_loss

        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        round_num = config.get("server_round", 0)
        proximal_mu = config.get("proximal_mu", 0.0)
        if should_print(round_num):
            print(f"[Client {self.client_id}] --- ROUND {round_num} (FIT) --- proximal_mu={proximal_mu}")

        # Decay the learning rate across rounds.
        new_lr = INITIAL_LR * (LR_DECAY_RATE ** round_num)
        model.optimizer.learning_rate.assign(new_lr)

        model.set_weights(parameters)
        # global_weights is the anchor point (start-of-round weights) the
        # proximal term measures drift against — must be captured from
        # trainable_variables specifically (not model.get_weights(), which
        # also includes non-trainable BatchNorm moving stats). Using
        # get_weights() here caused a shape mismatch: it and
        # model.trainable_variables have different lengths/order once
        # BatchNorm layers are present, so zip() paired mismatched tensors.
        global_weights = [v.numpy() for v in model.trainable_variables]

        for _ in range(LOCAL_EPOCHS):
            train_one_epoch_fedprox(
                model, x_train, y_train, class_weight_dict,
                global_weights, proximal_mu, batch_size=256
            )

        val_loss, _ = model.evaluate(x_test, y_test, verbose=0)
        y_val_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
        val_acc, val_prec, val_rec, val_f1 = compute_all_metrics(y_test, y_val_pred)

        if should_print(round_num):
            print(f"[Client {self.client_id}] LOCAL VAL   → loss: {val_loss:.4f} | "
                  f"accuracy: {val_acc:.4f} | precision(macro): {val_prec:.4f} | "
                  f"recall(macro): {val_rec:.4f} | f1(macro): {val_f1:.4f}")

        # SHAP on the fixed sample — rebuilt each round so it reflects the
        # freshly updated model weights.
        # NOTE: DeepExplainer does NOT reliably support LSTM layers — its
        # gradient-override mechanism doesn't know how to handle LSTM's
        # internal gate ops in TF2 eager mode (fails with
        # "'TFDeep' object has no attribute 'between_tensors'").
        # GradientExplainer uses a different technique (expected gradients
        # via input noise) that works with recurrent layers.
        #
        # COST NOTE: by default, GradientExplainer computes attributions for
        # EVERY output class, using `nsamples` interpolations per input.
        # With 34 classes, 500 samples, and the default nsamples=200, that's
        # ~34 * 500 * 200 ≈ 3.4M forward/backward passes through the model —
        # per client, per round. Since this SHAP output is only used for
        # aggregation weighting (not a precise per-class explanation), we
        # cut cost by (a) only explaining each sample's top-3 predicted
        # classes via ranked_outputs instead of all 34, and (b) using far
        # fewer interpolation samples. This trades some estimate precision
        # for a large speedup — tune ranked_outputs/nsamples up if runtime
        # allows and you want tighter estimates.
        explainer = shap.GradientExplainer(model, background)
        shap_values, _ = explainer.shap_values(x_sample, nsamples=30, ranked_outputs=3)

        shap_array = np.array(shap_values)                        # (classes, samples, timesteps, features, channels)
        mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 1))  # (timesteps, features, channels)
        mean_abs_shap = mean_abs_shap.mean(axis=(0, -1))          # collapse timesteps & channel -> (features,)

        feature_names = df.drop(columns=['Label']).columns.tolist()
        num_features = mean_abs_shap.shape[0]
        feature_names = feature_names[:num_features]

        shap_metrics = {feat: float(mean_abs_shap[i])
                         for i, feat in enumerate(feature_names)}

        return model.get_weights(), len(x_train), shap_metrics

    def evaluate(self, parameters, config):
        round_num = config.get("server_round", 0)
        if should_print(round_num):
            print(f"[Client {self.client_id}] --- ROUND {round_num} (EVALUATE) ---")
        model.set_weights(parameters)
        loss, _ = model.evaluate(x_test, y_test, verbose=0)
        y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
        accuracy, precision, recall, f1 = compute_all_metrics(y_test, y_pred)

        if should_print(round_num):
            print(f"[Client {self.client_id}] GLOBAL → loss: {loss:.4f} | accuracy: {accuracy:.4f} | "
                  f"precision(macro): {precision:.4f} | recall(macro): {recall:.4f} | f1(macro): {f1:.4f}")

        return loss, len(x_test), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


fl.client.start_numpy_client(
    server_address="localhost:" + str(sys.argv[1]),
    client=FlowerClient(client_id=CLIENT_ID),
    grpc_max_message_length=1024 * 1024 * 1024
)
