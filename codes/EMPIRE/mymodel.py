import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden1=16, hidden2=4, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # nn.Linear(hidden1, hidden2),
            # nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden1, 1)
        )

    def forward(self, x):
        return self.net(x)