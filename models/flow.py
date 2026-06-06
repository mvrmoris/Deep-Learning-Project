import torch.nn as nn

class FlowNet(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x):
        return self.net(x)

def flow_loss(model, x, y):
    target = y - x
    pred = model(x)
    return ((pred - target) ** 2).mean()
