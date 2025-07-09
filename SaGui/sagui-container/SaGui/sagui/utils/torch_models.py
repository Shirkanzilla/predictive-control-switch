from torch import nn

class ExpectedCostRegressor(nn.Module):
    def __init__(self):
        super(ExpectedCostRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(57, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,1),
        )
    def forward(self, x):
        return self.fc(x)

class ExpectedCostClassifier(nn.Module):
    def __init__(self):
        super(ExpectedCostClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(57, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)