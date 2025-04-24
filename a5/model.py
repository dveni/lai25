import torch.nn.functional as F
from torch import nn

class MyDummyLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, layer_idx: int):
        super(MyDummyLayer, self).__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.layer_idx = layer_idx

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class MyDummyModel(nn.Module):
    def __init__(self, number_of_layers: int, hidden_size: int, intermediate_size: int):
        super(MyDummyModel, self).__init__()
        self.layers = nn.ModuleList([MyDummyLayer(hidden_size, intermediate_size, layer_idx = i) for i in range(number_of_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x