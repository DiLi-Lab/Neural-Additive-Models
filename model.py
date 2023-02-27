from datetime import time

import torch
from torch.nn import Module, Conv1d, MaxPool1d, Dropout, Flatten, Linear
from torch.nn import functional


class Model(Module):

    def __init__(self):
        super().__init__()
        self.startTime = None
        self.conv1 = Conv1d(3, 96, 6, groups=3)
        self.dropout1 = Dropout(0.2)
        self.max_pool1 = MaxPool1d(12, stride=6)

        self.conv2 = Conv1d(96, 96, 6, groups=3)
        self.dropout2 = Dropout(0.3)
        self.max_pool2 = MaxPool1d(12, stride=6)

        self.linear1 = Linear(2496, 712)

        self.linear2 = Linear(712, 1)

    def forward(self, features) -> torch.Tensor:
        x = self.dropout1(features)
        x = functional.relu(self.conv1(x))
        x = self.max_pool1(x)

        x = self.dropout2(x)
        x = functional.relu(self.conv2(x))
        x = self.max_pool2(x)

        x = Flatten(x)

        x = functional.relu(self.linear1(x))
        x = functional.sigmoid(self.linear2(x))

        return x




