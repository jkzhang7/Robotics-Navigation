from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

class rmpDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['img_name'][idx]
        total = np.load('./data/rmp_label/' + str(img_path))
        goal = torch.as_tensor(total.item()['goals_local'], dtype=torch.float32)
        post = torch.as_tensor(total.item()['pos'], dtype=torch.float32)
        heading = torch.as_tensor(total.item()['heading'], dtype=torch.float32)
        time_step = torch.as_tensor(total.item()['time_step'], dtype=torch.float32)
        cp_metrics = torch.as_tensor(total.item()['control_point_metrics'], dtype=torch.float32)
        cp_accels = torch.as_tensor(total.item()['control_point_accels'], dtype=torch.float32)
        velocity = torch.as_tensor(total.item()['velocity'], dtype=torch.float32)
        angular_v = torch.as_tensor(total.item()['angular_velocity'], dtype=torch.float32)
        frame = torch.as_tensor((total.item()['frame']))
        frame = frame.permute(2, 0, 1)
        cp_accels = cp_accels.view(1, 12*2).squeeze(0)
        cp_metrics = cp_metrics.view(1, 12*2*2).squeeze(0)
        return [frame, velocity, angular_v, goal, cp_metrics, cp_accels]


def loadData(csv_path):
    df_dataset = pd.read_csv(csv_path)
    df_train, df_test = train_test_split(df_dataset, test_size=0.2)
    df_train.index = range(len(df_train))
    df_test.index = range(len(df_test))

    train_data = rmpDataset(df_train)
    test_data = rmpDataset(df_test)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=2)

    return train_data, test_data, train_loader, test_loader
