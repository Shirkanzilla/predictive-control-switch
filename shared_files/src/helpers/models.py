import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dimension, hidden_dimension=[64], classification=True):
        super(NeuralNetwork, self).__init__()
        layers = []
        if len(hidden_dimension) == 0:
            layers.append(nn.Linear(input_dimension, 1))
        else:
            for i, out_features in enumerate(hidden_dimension):
                if i == 0:
                    layers.append(nn.Linear(input_dimension, out_features))
                else:
                    layers.append(nn.Linear(hidden_dimension[i-1], out_features))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dimension[-1], 1))
        self.model = nn.Sequential(*layers)
        if classification:
            self.model.add_module("sigmoid", nn.Sigmoid())
    
    def forward(self, x):
        return self.model(x)