import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):

    def __init__(self, in_features=40, hidden_features=30, hidden_count=3, nclasses=10):
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            *[nn.Linear(hidden_features, hidden_features) for _ in range(hidden_count)],
            nn.Linear(hidden_features, nclasses),
        )

    def forward(self, x):
        return self.model(x)
