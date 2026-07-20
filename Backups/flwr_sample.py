from flwr.client import NumPyClient, ClientApp
from flwr.common import ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from torchvision import datasets
from torch.utils.data import random_split
from collections import OrderedDict
from flwr.common.logger import log, INFO

from utils2 import *

# -----------------------------
# Dataset Preparation
# -----------------------------
trainset = datasets.MNIST("./MNIST_data/", download=True, train=True, transform=transform)

total_length = len(trainset)
split_size = total_length // 3

torch.manual_seed(42)
part1, part2, part3 = random_split(trainset, [split_size] * 3)

part1 = exclude_digits(part1, excluded_digits=[1, 3, 7])
part2 = exclude_digits(part2, excluded_digits=[2, 5, 8])
part3 = exclude_digits(part3, excluded_digits=[4, 6, 9])

train_sets = [part1, part2, part3]

testset = datasets.MNIST("./MNIST_data/", download=True, train=False, transform=transform)
testset_137 = include_digits(testset, [1, 3, 7])
testset_258 = include_digits(testset, [2, 5, 8])
testset_469 = include_digits(testset, [4, 6, 9])

# -----------------------------
# Helper Functions
# -----------------------------
def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# -----------------------------
# Flower Client
# -----------------------------
class FlowerClient(NumPyClient):
    def __init__(self, net, trainset, testset):
        self.net = net
        self.trainset = trainset
        self.testset = testset

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_model(self.net, self.trainset)
        return get_weights(self.net), len(self.trainset), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate_model(self.net, self.testset)
        return loss, len(self.testset), {"accuracy": accuracy}

# Correct indentation: client_fn must be OUTSIDE the class
def client_fn(context: Context):
    net = SimpleModel()
    partition_id = int(context.node_config["partition-id"])
    return FlowerClient(net, train_sets[partition_id], testset).to_client()

# -----------------------------
# Server Evaluation Function
# -----------------------------
def evaluate(server_round, parameters, config):
    net = SimpleModel()
    set_weights(net, parameters)

    _, accuracy = evaluate_model(net, testset)
    _, acc137 = evaluate_model(net, testset_137)
    _, acc258 = evaluate_model(net, testset_258)
    _, acc469 = evaluate_model(net, testset_469)

    print(f"Accuracy (All): {accuracy:.4f}")
    print(f"Accuracy [1,3,7]: {acc137:.4f}")
    print(f"Accuracy [2,5,8]: {acc258:.4f}")
    print(f"Accuracy [4,6,9]: {acc469:.4f}")

    if server_round == 3:
        cm = compute_confusion_matrix(net, testset)
        plot_confusion_matrix(cm, "Final Global Model")

# Initialize global model parameters
net = SimpleModel()
params = ndarrays_to_parameters(get_weights(net))

# -----------------------------
# Server Function
# -----------------------------
def server_fn(context: Context):
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        initial_parameters=params,
        evaluate_fn=evaluate,
    )
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)

# -----------------------------
# Run Simulation
# -----------------------------
server = ServerApp(server_fn=server_fn)
client = ClientApp(client_fn)

run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=3,
    backend_config=backend_setup,
)