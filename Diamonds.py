import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

df = pd.read_csv('https://query.data.world/s/iegeupnvgat3xczjxgo3lr76r5wf3m')
print(df.columns)
print(len(df.carat))

class CustomDataset(Dataset):
    def __init__(self,df):
        self.x = df.iloc[]



