import torch
import torch.nn as nn


class SimpleClassifier(nn.Sequential):

    def __init__(self, in_channels=40, hidden_channels=[30], n_classes=10, bias=True, actfn=nn.LeakyReLU):
        if hidden_channels is None:
            hidden_channels = []
        if not isinstance(hidden_channels, (tuple, list)):
            hidden_channels = [hidden_channels]

        ins = [in_channels, *hidden_channels]
        outs = [*hidden_channels, n_classes]
        linears = [nn.Linear(i, o, bias) for i, o in zip(ins, outs)]

        layers = []
        for l in linears:
            layers.append(l)
            layers.append(actfn())

        super().__init__(*layers)
