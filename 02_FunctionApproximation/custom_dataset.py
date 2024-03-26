import pandas as pd
import numpy as np
from typing import Literal
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):
    def __init__(self, data_dir, data_mode:Literal['train','val','test']='train', transform=None, label_transform=None):
        self.data = pd.read_csv(data_dir, header=None, delimiter=';')
        self.data = pd.DataFrame(StandardScaler().fit_transform(self.data))
        self.data_mode = data_mode
        self.transform = transform
        self.label_transform = label_transform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.data_mode == 'train':
            value_data = self.data.iloc[index, 0]
            value_label = self.data.iloc[index, 1]
        if self.data_mode == 'test':
            value_data = self.data.iloc[index, 0]
        
        if self.data_mode == 'train':
            value_data = torch.tensor([value_data]).float()
            value_label = torch.tensor([value_label]).float()
        else:
            value_label = torch.tensor([value_label])

        if self.data_mode == 'train':
            value_data, value_label = value_data.to(self.device), value_label.to(self.device)
        else:
            value_data, value_label = value_data.to(self.device), None

        return value_data, value_label
