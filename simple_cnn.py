from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
    
dataload = CustomDataset('/Users/jonah.krop/Documents/USC/usc_dsci_565_project/data/tensors.pt')
data = DataLoader(dataload, batch_size=16, shuffle=True)

import pdb; pdb.set_trace()

print('done')