import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# -----------------------------
# Transform for MNIST
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# -----------------------------
# Simple CNN Model
# -----------------------------
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -----------------------------
# Training Function
# -----------------------------
def train_model(model, dataset, epochs=1, batch_size=32, lr=0.001):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model, dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    correct, total, loss_sum = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return loss_sum / len(loader), accuracy

# -----------------------------
# Digit Filtering
# -----------------------------
def exclude_digits(dataset, excluded_digits):
    indices = [i for i, (_, label) in enumerate(dataset) if label not in excluded_digits]
    return Subset(dataset, indices)

def include_digits(dataset, included_digits):
    indices = [i for i, (_, label) in enumerate(dataset) if label in included_digits]
    return Subset(dataset, indices)

# -----------------------------
# Confusion Matrix
# -----------------------------
def compute_confusion_matrix(model, dataset):
    loader = DataLoader(dataset, batch_size=64)
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return confusion_matrix(all_labels, all_preds)

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# -----------------------------
# Flower Simulation Backend
# -----------------------------
backend_setup = {
    "client_resources": {"num_cpus": 1},
    "server_resources": {"num_cpus": 1},
}