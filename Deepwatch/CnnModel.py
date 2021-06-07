import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

def weights_init(m):
  weight_shape = list(m.weight.data.size())
  fan_in = np.prod(weight_shape[1:4])
  fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
  w_bound = np.sqrt(6. / (fan_in + fan_out))
  m.weight.data.uniform_(-w_bound, w_bound)
  m.bias.data.fill_(0)

class CnnModel(nn.Module):
  def __init__(self):
    super(CnnModel, self).__init__()
    # Convolution 1
    self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
    self.relu1 = nn.ReLU()

    # Max pool 1
    self.maxpool1 = nn.MaxPool2d(kernel_size=2)

    # Convolution 2
    self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
    self.relu2 = nn.ReLU()

    # Max pool 2
    self.maxpool2 = nn.MaxPool2d(kernel_size=2)

    # Fully connected 1 (readout)
    self.fc1 = nn.Linear(32 * 7 * 7, 10)  


  def forward(self, x):
    out = self.cnn1(x)
    out = self.relu1(out)

    out = self.maxpool1(out)

    out = self.conv2(out)
    out = self.relu2(out)

    out = self.maxpool2(out)

    out = out.view(out.size(0), -1)

    out = self.fc1(out)


model = CnnModel()

print(model.parameters)