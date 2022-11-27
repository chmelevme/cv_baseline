import torch.utils.data as data
import os
import numpy as np
import torch


class CustomDataset(data.Dataset):
    def __init__(self, split, transform=None, preload_data=False):
        super(CustomDataset, self).__init__()
        self.feat_path = os.path.join('data', 'processed', f'{split}', 'feat')
        self.mask_path = os.path.join('data', 'processed', f'{split}', 'mask')
        self.data = os.listdir(self.feat_path)

    def __getitem__(self, index):
        input_feat = np.load(os.path.join(self.feat_path, self.data[index]), allow_pickle=True)
        input_target = np.load(os.path.join(self.mask_path, self.data[index]), allow_pickle=True)
        input = torch.tensor(input_feat, dtype=torch.float).unsqueeze(dim=0)
        target = torch.tensor(input_target, dtype=torch.long)

        return input, target

    def __len__(self):
        return len(self.data)


def make_train_val_dataloader():
    train_df = CustomDataset('train')
    valid_df = CustomDataset('test')
    train_dataloader = data.DataLoader(train_df, 1)
    valid_dataloader = data.DataLoader(valid_df, 1)
    return train_dataloader, valid_dataloader
