import json
import numpy as np
import random
import time

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from datasets import CNNTrainingDataset
from helpers import EarlyStopper, step
from models import CNNSurfaceClassifier

device = "cuda:0" if torch.cuda.is_available() else "cpu"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

BATCH_SIZE = 32
INPUT_SIZE = 6
NUM_EPOCHS = 100
DATA_DIR = Path('../data/train_set/csv/')
LOOKBACK = 8/3
SAMPLING_FREQUENCY = 75.
DATASET_FREQUENCY = 150.
SUBSET = ('imu',)
CONFIGURATIONS = ('4W',)

with open('../data/train_set/labels.json') as fp:
    labels = json.load(fp)

dataset = [(DATA_DIR.joinpath(key + '.csv'), values['surface'])
           for key, values in labels.items()
           if values['kinematics'] in CONFIGURATIONS and values['spacing'] == 'R1' and 'T1' in values['trajectory']]

X = [run[0] for run in dataset]
y_primary = [run[1] for run in dataset]

# y_secondary = []
# y_secondary = ['slippery' if label in ('1_Panele', '5_Spienione_PCV', '6_Linoleum')
#                else 'grippy' if label in ('3_Wykladzina_jasna', '8_Pusta_plyta', '9_podklady')
#                else 'neutral' for label in y_primary]
y_secondary = ['slippery' if label in ('3_Wykladzina_jasna', '4_Trawa')
               else 'grippy' if label in ('5_Spienione_PCV', '8_Pusta_plyta', '9_podklady', '10_Mata_ukladana')
               else 'neutral' for label in y_primary]   # Pawel set
# y_secondary = ['slippery' if label in ('3_Wykladzina_jasna', '4_Trawa')
#                else 'grippy' if label in ('2_Wykladzina_czarna', '5_Spienione_PCV', '9_podklady', '10_Mata_ukladana')
#                else 'neutral' for label in y_primary] # Clustering set

lb = LabelBinarizer()
if y_secondary:
    lb.fit(y_secondary)
    y = lb.transform(y_secondary)
else:
    lb.fit(y_primary)
    y = lb.transform(y_primary)
classes = lb.classes_
num_classes = len(classes)
y = y.reshape(-1, num_classes)

X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, stratify=y_training)
X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

train_dataloader = DataLoader(
    CNNTrainingDataset(
        X_train,
        y_train,
        sample_freq=SAMPLING_FREQUENCY,
        data_freq=DATASET_FREQUENCY,
        lookback=LOOKBACK,
        subset=SUBSET,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_dataloader = DataLoader(
    CNNTrainingDataset(
        X_val,
        y_val,
        sample_freq=SAMPLING_FREQUENCY,
        data_freq=DATASET_FREQUENCY,
        lookback=LOOKBACK,
        subset=SUBSET,
    ),
    batch_size=BATCH_SIZE,
)
test_dataloader = DataLoader(
    CNNTrainingDataset(
        X_test,
        y_test,
        sample_freq=SAMPLING_FREQUENCY,
        data_freq=DATASET_FREQUENCY,
        lookback=LOOKBACK,
        subset=SUBSET,
    ),
    batch_size=BATCH_SIZE,
)

cnn_model = CNNSurfaceClassifier(input_size=INPUT_SIZE, output_size=num_classes).to(device)
best_model = cnn_model.state_dict()

optimizer = torch.optim.Adam(
    cnn_model.parameters(),
    lr=1e-3,
    eps=1e-6,
    weight_decay=1e-3,
    )

scheduler = ExponentialLR(optimizer, gamma=0.9)

early_stopper = EarlyStopper()

criterion = nn.CrossEntropyLoss()

train_batches = len(train_dataloader)
val_batches = len(val_dataloader)

for epoch in range(NUM_EPOCHS):
    running_train_loss = 0.0
    running_val_loss = 0.0

    pbar = tqdm(train_dataloader, total=train_batches)
    cnn_model.train()
    for idx, train_batch in enumerate(pbar):
        train_loss, _ = step(cnn_model, train_batch, criterion, device, train=True, optimizer=optimizer)
        running_train_loss += train_loss

        pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Training loss: {running_train_loss / (idx + 1):.2E}")
    scheduler.step()

    pbar_val = tqdm(val_dataloader, total=val_batches)
    cnn_model.eval()
    with torch.no_grad():
        for idx, val_batch in enumerate(pbar_val):
            val_loss, _ = step(cnn_model, val_batch, criterion, device)
            running_val_loss += val_loss

            pbar_val.set_description(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}, Validation loss: {running_val_loss / (idx + 1):.2E}")

    validation_loss = running_val_loss / (idx + 1)
    if early_stopper.early_stop(validation_loss):
        print(f"Split ended on epoch {epoch + 1 - early_stopper.patience}!")
        break
    if early_stopper.counter == 0:
        best_model = cnn_model.state_dict()

model_name = 'cnn_classifier_' + '_'.join((str(num_classes),) + CONFIGURATIONS + SUBSET) + '_' + time.strftime(
    "%Y-%m-%d-%H-%M-%S")
torch.save(best_model, f"../results/checkpoints/{model_name}.pt")

cnn_model = CNNSurfaceClassifier(input_size=INPUT_SIZE, output_size=num_classes).to(device)
cnn_model.load_state_dict(torch.load(f"../results/checkpoints/{model_name}.pt"))

test_batches = len(test_dataloader)
y_true, y_pred = [], []
running_test_loss = 0.0

pbar_test = tqdm(test_dataloader, total=test_batches)
cnn_model.eval()
with torch.no_grad():
    for idx, test_batch in enumerate(pbar_test):
        test_loss, test_outputs = step(cnn_model, test_batch, criterion, device)
        running_test_loss += test_loss

        y_true.extend(torch.argmax(test_batch[1], dim=1).cpu().numpy())
        y_pred.extend(torch.argmax(test_outputs, dim=1).cpu().numpy())

        pbar_test.set_description(f"Test loss: {running_test_loss / (idx + 1):.2E}")
