import torch
import torch.nn as nn
import torch.nn.functional as F
from .splittable_layers import SplittableLinear

class NetworkModel(nn.Module):
    def __init__(self, without_rel, dims, alpha, beta, goodness_of_fit_cutoff):
        super().__init__()
        self.without_rel = without_rel
        self.dims = dims
        self.fc = nn.ModuleList()

        current_dim = dims[0]
        for i in range(1, len(dims)):
            layer = SplittableLinear(
                current_dim,
                dims[i],
                alpha=alpha,
                beta=beta,
                goodness_of_fit_cutoff=goodness_of_fit_cutoff[0],
                name=f'layer {i}'
            )
            self.fc.append(layer)
            current_dim = dims[i]

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Modified to handle MNIST data properly
        for j, layer in enumerate(self.fc):
            x = layer(x)
            if not self.without_rel[j]:
                x = F.relu(x)
        x = F.log_softmax(x, dim=1)
        return x

    def get_layer_matrix(self, index):
        layer_matrix = self.fc[index].weight.cpu().detach().numpy()
        return layer_matrix

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())