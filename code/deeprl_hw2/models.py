import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearDQN(nn.Module):
    def __init__(self, action_num):
        super(LinearDQN, self).__init__()
        self.linear = nn.Linear(84 * 84 * 4, action_num)

    def forward(self, x):
        return self.linear(x)


class DeepDQN(nn.Module):
    def __init__(self, action_num):
        super(DeepDQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, action_num)

    def forward(self, x):
        x = x.reshape(-1, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.reshape(x.size(0), -1)))
        return self.fc2(x)