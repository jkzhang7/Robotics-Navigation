import torch
import multiprocessing
from data.dataset import rmpDataset, loadData
from models.net import neuralRMP
from torch import nn
import torch.nn.functional as F
import numpy as np

BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
EPOCHS = 50
LEARNING_RATE = 0.001
MOMENTUM = 0.9
USE_CUDA = True
SEED = 0
PRINT_INTERVAL = 100
WEIGHT_DECAY = 0.0005

use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
print('num cpus:', multiprocessing.cpu_count())

csv_path = './data/img_name.csv'
train_data, test_data, train_loader, test_loader = loadData(csv_path)

# print(train_data[])
model = neuralRMP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train():
    for epoch in range(EPOCHS):
        # pass
        running_loss = 0
        running_acc = 0
        print('Running Epoch: ', epoch)
        for i, data in enumerate(train_loader):
            frame, velocity, angular_v, goal, cp_metrics, cp_accels = data
            optimizer.zero_grad()

            accel, metric_full = model(frame, velocity, angular_v, goal)
            loss = criterion(accel, cp_accels) + criterion(metric_full, cp_metrics)
            loss.backward()
            optimizer.step()

            running_loss += loss
            running_acc += (torch.sum(accel == cp_accels) + torch.sum(metric_full == cp_metrics))

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_loss / len(train_loader.dataset)
        print('loss: ', epoch_loss, 'acc: ', epoch_acc)

train()
# def test():

