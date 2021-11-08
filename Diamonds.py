import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.nn import functional

#Learning the syntax

# df = pd.read_csv('Diamonds.csv')
# size = df.size
# print(df.columns)
# print(torch.tensor(df.price))
# t1 = torch.tensor(df[["carat", "price"]].values)
# print(t1[3])
#
# cat = {'Ideal': 0, 'Premium': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4}
# df["cut"] = df["cut"].map(cat)
# print(torch.tensor(df.cut))
#
# tensor1 = torch.nn.functional.one_hot(torch.tensor(df.cut))
# tensor2 = torch.cat((tensor1, t1), 1)
#
# t3 = torch.cat((torch.tensor(df[["carat"]].values), tensor1,
#       torch.tensor(df[["depth", "table", "x", "y", "z"]].values)), 1)


class CustomDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        cut_cat = {'Ideal': 0, 'Premium': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4}
        df["cut"] = df["cut"].map(cut_cat)
        one_hot_cut = torch.nn.functional.one_hot(torch.tensor(df.cut))

        color_cat = {'G': 0, 'E': 1, 'F': 2, 'H': 3, 'D': 4, 'I': 5, 'J': 6}
        df["color"] = df["color"].map(color_cat)
        one_hot_color = torch.nn.functional.one_hot(torch.tensor(df.color))

        clarity_cat = {'SI1': 0, 'VS2': 1, 'SI2': 2, 'VS1': 3, 'VVS2': 4, 'VVS1': 5, 'IF': 6, 'I1': 7}
        df["clarity"] = df["clarity"].map(clarity_cat)
        one_hot_clarity = torch.nn.functional.one_hot(torch.tensor(df.clarity))

        self.features = torch.cat((torch.tensor(df[["carat"]].values), one_hot_cut, one_hot_color, one_hot_clarity,
                                   torch.tensor(df[["depth", "table", "x", "y", "z"]].values)), 1)
        self.price = torch.tensor(df.price)

    def __len__(self):
        return len(self.price)

    def __getitem__(self, idx):
        return self.features[idx], self.price[idx]


dataset = CustomDataset('Diamonds.csv')
print("Num Diamonds in Dataset:", len(dataset))
print("One example from dataset:", dataset[2])

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=16)
val_loader = DataLoader(dataset=val_dataset, batch_size=16)



