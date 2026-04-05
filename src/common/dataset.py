import torch
from torch.utils.data import Dataset
import pandas as pd

class StarbucksDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.loc[:, self.data.std() > 0]
        self.labels = self.data["completed"].values
        self.data.drop("completed", axis=1, inplace=True)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        result = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            result = self.transform(result)
        return result, label