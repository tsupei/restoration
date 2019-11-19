import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


# FFNN
class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_of_class = args['class-number']
        num_of_features = args['hidden-dimension']

        # fully-connected layer
        self.fc = nn.Linear(num_of_features, num_of_class)
        self.dropout = nn.Dropout(args["dropout-rate"])

    def forward(self, x):
        x = x.squeeze(1)
        x = self.dropout(x)
        logit = self.fc(x)
        return logit
