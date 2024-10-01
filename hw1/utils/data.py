import numpy as np
import glob
from torch.utils.data import Dataset
import torch
import json
from pathlib import Path

class SlakhDataset(Dataset):
    def __init__(self, datasplit, folder='./slakh'):
        '''
        :param datasplit: (train, validation, test)
        '''
        assert datasplit in ['train', 'validation', 'test']
        self.datasplit = datasplit
        self.files = glob.glob(f'{folder}/{datasplit}/*.npy')
        self.jsonfiles = open(f"{folder}/{datasplit}_labels.json", "r")
        self.all_labels = json.load(self.jsonfiles)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(self.files[idx])
        labels = self.all_labels[Path(file).name]

        return torch.tensor(data), torch.tensor(labels)

if __name__ == '__main__':
    dataset = SlakhDataset(datasplit='train')
    for i in range(len(dataset)):
        data, labels = dataset.__getitem__(i)
    print(data.shape, labels)









