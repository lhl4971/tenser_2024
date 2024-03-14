import os
import pandas as pd
from PIL import Image
from typing import Literal
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data_dir, label_dir=None, data_mode:Literal['train','val','test']='train', transform=None, label_transform=None):
        self.data_dir = data_dir
        if label_dir:
            self.data_label = pd.read_csv(label_dir)
        else:
            label_list = os.listdir(self.data_dir)
            label_list = [os.path.splitext(item)[0] for item in label_list]
            label_list = [int(item) for item in label_list]
            self.data_label = sorted(label_list)
        self.data_mode = data_mode
        self.transform = transform
        self.label_transform = label_transform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, index):
        if self.data_mode == 'train':
            img_name = str(self.data_label.iloc[index, 0]) + '.jpg'
            img_path = os.path.join(self.data_dir, img_name)
            image = Image.open(img_path)
            label = self.data_label.iloc[index, 1]
        if self.data_mode == 'test':
            img_name = str(self.data_label[index]) + '.jpg'
            img_path = os.path.join(self.data_dir, img_name)
            image = Image.open(img_path)
            label = img_name
        
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        if self.data_mode == 'train':
            image, label = image.to(self.device), torch.tensor(label).to(self.device)
        else:
            image, label = image.to(self.device), os.path.splitext(label)[0]

        return image, label
