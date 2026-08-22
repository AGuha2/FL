# FedSHAPBlend

This repository contains the implementation of the FedSHAPBlend federated learning framework for IoT intrusion detection using the CICIoT2023 dataset.
https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/

## Files

- `create_6class_balanced_specialist_split.py`  
  Prepares the CICIoT2023 dataset. It performs label mapping, class balancing, creates the server holdout, creates the five non-IID client datasets, fits the scaler, and saves distribution summaries.

- `Labels6.py`  
  Maps the original CICIoT2023 attack labels into six traffic classes:
  - BENIGN
  - DDOS
  - DOS
  - MIRAI
  - RECON
  - OTHER

  It also defines the label encoder, number of classes, and number of input features.

- `model.py`  
  Contains the model definition currently imported by the server and client files.

- `model_mlp.py`  
  Contains the MLP architecture used in the MLP experiments.

- `model_cnn.py`  
  Contains the CNN architecture used in the CNN experiments.

- `model_lstm.py`  
  Contains the CNN-LSTM architecture used in the CNN-LSTM experiments.

- `server.py`  
  Runs the Flower federated learning server.

  It handles:
  - communication with the five clients;
  - collection of local model updates;
  - server-validation evaluation;
  - macro F1-based reliability calculation;
  - SHAP alignment calculation;
  - FedAvg/FedSHAPBlend aggregation;
  - global model evaluation;
  - saving metrics and confusion matrices;
  - final-round SHAP analysis.

  The aggregation configuration is controlled using:

  ```python
  SHAP_BLEND = 0.0
  ```
- `client1.py`  
  Runs Client 1. 

- `client2.py`  
  Runs Client 2.

- `client3.py`  
  Runs Client 3.

- `client4.py`  
  Runs Client 4.

- `client5.py`  
  Runs Client 5.

All five clients perform local training, calculate local metrics and SHAP feature-importance values, and return their model parameters to the server.

---

## Running Order

### 1. Prepare the Dataset

Update the CICIoT2023 source and output paths inside:

```text
create_6class_balanced_specialist_split.py
```

Then run:

```bash
python create_6class_balanced_specialist_split.py
```

This generates:

- server dataset;
- Client 1 dataset;
- Client 2 dataset;
- Client 3 dataset;
- Client 4 dataset;
- Client 5 dataset;
- fitted RobustScaler;
- distribution summary files.

---

### 2. Update the Data Paths

The scripts currently contain local paths such as:

```text
D:\CAPSTONE\
```

Update the paths inside:

```text
server.py
client1.py
client2.py
client3.py
client4.py
client5.py
```

so they point to the generated datasets and scaler on your machine.

---

### 3. Select the Model

Choose one model architecture:

```text
model_mlp.py
model_cnn.py
model_lstm.py
```

The server and all five clients must use the same architecture.

The current files import:

```python
from model import build_model
```

Therefore, either place the required architecture in `model.py` or change the import consistently.

For MLP:

```python
from model_mlp import build_model
```

For CNN:

```python
from model_cnn import build_model
```

For CNN-LSTM:

```python
from model_lstm import build_model
```

---

### 4. Select the Aggregation Method

Open `server.py`.

For FedAvg:

```python
SHAP_BLEND = 0.0
```

For Hybrid FedSHAPBlend:

```python
SHAP_BLEND = 0.5
```

For Pure FedSHAPBlend:

```python
SHAP_BLEND = 1.0
```

---

### 5. Start the Server

Open a terminal and run:

```bash
python server.py 8080
```

`8080` is the communication port. Another available port may also be used.

Keep the server terminal running.

---

### 6. Start Client 1

Open another terminal:

```bash
python client1.py 8080
```

---

### 7. Start Client 2

Open another terminal:

```bash
python client2.py 8080
```

---

### 8. Start Client 3

Open another terminal:

```bash
python client3.py 8080
```

---

### 9. Start Client 4

Open another terminal:

```bash
python client4.py 8080
```

---

### 10. Start Client 5

Open another terminal:

```bash
python client5.py 8080
```

The server and all five clients must use the same port.

A complete experiment therefore uses six terminals:

```text
Terminal 1:
python server.py 8080

Terminal 2:
python client1.py 8080

Terminal 3:
python client2.py 8080

Terminal 4:
python client3.py 8080

Terminal 5:
python client4.py 8080

Terminal 6:
python client5.py 8080
```

The federated experiment runs for 10 communication rounds.

---

## Experiment Order

Run three aggregation configurations for each model architecture.

### MLP

```text
1. MLP + FedAvg
   SHAP_BLEND = 0.0

2. MLP + Hybrid FedSHAPBlend
   SHAP_BLEND = 0.5

3. MLP + Pure FedSHAPBlend
   SHAP_BLEND = 1.0
```

### CNN

```text
4. CNN + FedAvg
   SHAP_BLEND = 0.0

5. CNN + Hybrid FedSHAPBlend
   SHAP_BLEND = 0.5

6. CNN + Pure FedSHAPBlend
   SHAP_BLEND = 1.0
```

### CNN-LSTM

```text
7. CNN-LSTM + FedAvg
   SHAP_BLEND = 0.0

8. CNN-LSTM + Hybrid FedSHAPBlend
   SHAP_BLEND = 0.5

9. CNN-LSTM + Pure FedSHAPBlend
   SHAP_BLEND = 1.0
```

For each experiment:

```text
Select model
    ↓
Set SHAP_BLEND
    ↓
Start server
    ↓
Start Clients 1-5
    ↓
Run 10 federated rounds
    ↓
Save results
```

Restart the server and all five clients before starting the next experiment.
