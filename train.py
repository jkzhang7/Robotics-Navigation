import torch
import multiprocessing
from data.dataset import rmpDataset, loadData
from models.net import neuralRMP
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1
EPOCHS = 15
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
train_data, test_data, train_loader, test_loader = loadData(csv_path, BATCH_SIZE, 0)

model = neuralRMP().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
epoch_list = [_ for _ in range(EPOCHS)]


def train():
    loss_list = []
    for epoch in range(EPOCHS):
        # pass
        running_loss = 0
        running_acc = 0
        print('Running Epoch: ', epoch)
        for i, data in enumerate(train_loader):
            frame, velocity, angular_v, goal, cp_metrics, cp_accels = data
            # print(frame.shape)
            optimizer.zero_grad()
            frame = frame.to(device)
            velocity = velocity.to(device)
            angular_v = angular_v.to(device)
            goal = goal.to(device)
            cp_metrics = cp_metrics.to(device)
            cp_accels = cp_accels.to(device)
            accel, metric_full = model(frame, velocity, angular_v, goal)
            loss = criterion(accel, cp_accels) + criterion(metric_full, cp_metrics)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += (torch.sum(accel == cp_accels) + torch.sum(metric_full == cp_metrics))
            if i % 100 == 0:
                print('Running', i, '/', len(train_loader.dataset))

        epoch_loss = running_loss / len(train_loader.dataset)
        loss_list.append(epoch_loss)
        epoch_acc = running_acc / len(train_loader.dataset)
        print('epoch loss: ', float(epoch_loss), 'epoch acc: ', float(epoch_acc))
    plt.figure(1)
    plt.plot(epoch_list, loss_list)
    plt.savefig('trainloss.png')


train()

torch.save(model.state_dict(), './thirdRMP.pt')


def test():
    test_acc, test_loss = 0, 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            frame, velocity, angular_v, goal, cp_metrics, cp_accels = data
            optimizer.zero_grad()
            frame = frame.to(device)
            velocity = velocity.to(device)
            angular_v = angular_v.to(device)
            goal = goal.to(device)
            cp_metrics = cp_metrics.to(device)
            cp_accels = cp_accels.to(device)
            accel, metric_full = model(frame, velocity, angular_v, goal)
            loss = criterion(accel, cp_accels) + criterion(metric_full, cp_metrics)
            test_loss += loss
            test_acc += (torch.sum(accel == cp_accels) + torch.sum(metric_full == cp_metrics))
    test_loss = float(test_loss) / len(test_loader.dataset)
    test_acc = float(test_acc) / len(test_loader.dataset)

    print('Test loss: ', test_loss, 'Test acc: ', test_acc)


test()
