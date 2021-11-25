'''
source: https://discuss.pytorch.org/t/shuffling-of-the-dataset/36166/5
'''

import torch
class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 11
    def __getitem__(self, idx):
        return idx

s=Dataset()
loader= torch.utils.data.DataLoader(s,
                                     batch_size=1, shuffle=True,
                                     num_workers=0)

for i in iter(loader):
    print(i)
