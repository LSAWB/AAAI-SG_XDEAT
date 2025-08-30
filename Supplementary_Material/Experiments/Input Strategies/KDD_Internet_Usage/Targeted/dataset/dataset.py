import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset): 
    def __init__(self, cat_data, num_data, T_cat_data, labels):
        
        # cat_data
        if isinstance(cat_data, torch.Tensor):
            self.cat_data = cat_data
        elif cat_data is None or len(cat_data) == 0:
            self.cat_data = torch.empty((len(labels), 0), dtype=torch.long)
        else:
            self.cat_data = torch.tensor(cat_data.values, dtype=torch.long)

        # num_data
        if isinstance(num_data, torch.Tensor):
            self.num_data = num_data
        elif num_data is None or len(num_data) == 0:
            self.num_data = torch.empty((len(labels), 0), dtype=torch.float32)
        else:
            self.num_data = torch.tensor(num_data.values, dtype=torch.float32)

        # ( Tree ) cat_data
        if isinstance(T_cat_data, torch.Tensor):
            self.T_cat_data = T_cat_data
        elif T_cat_data is None or len(T_cat_data) == 0:
            self.T_cat_data = torch.empty((len(labels), 0), dtype=torch.float32)
        else:
            self.T_cat_data = torch.tensor(T_cat_data.values, dtype=torch.float32)

        # Label
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cat_data[idx], self.num_data[idx], self.T_cat_data[idx], self.labels[idx]
