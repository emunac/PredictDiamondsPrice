import pandas as pd
import torch
from torch import nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.nn import functional
from tqdm import tqdm


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

        price = torch.tensor(df.price).float()
        #standart dev
        # mean = price.mean()
        # std = price.std()
        # self.price = (price - mean)/std
        self.price = price.log()

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

train_loader = DataLoader(dataset=train_dataset, batch_size=128)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)

model = nn.Linear(26, 1)
print(model.state_dict())

lr = 0.0001
n_epochs = 1000

loss_fn = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    model.train()
    losses = []
    for feat_batch, price_batch in tqdm(train_loader):

        yhat:torch.Tensor = model(feat_batch.float())

        loss = loss_fn(price_batch.float(), yhat.squeeze())
        loss:torch.Tensor
        #print(loss.item())
        losses.append(loss.detach())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for feat_val, price_val in val_loader:
            yhat_val: torch.Tensor = model(feat_val.float())
            val_loss = loss_fn(price_val.float(), yhat_val.squeeze())
            val_losses.append(val_loss.detach())


    print(epoch, "loss_train:", torch.stack(losses).mean(), "loss_val:", torch.stack(val_losses).mean())

print(model.state_dict())

#val_loss on a constant model
val_losses = []
with torch.no_grad():
    for feat_val, price_val in val_loader:
        #constant of log(mean(price))
        yhat_val = (3932.8 * torch.ones(len(price_val))).log()
        val_loss = loss_fn(price_val.float(), yhat_val.squeeze())
        val_losses.append(val_loss.detach())
print("val loss constant:", torch.stack(val_losses).mean())


