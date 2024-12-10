import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy.integrate import quad
from TracyWidom import TracyWidom

def mp_density(ndf, pdim, var=1):
    gamma = ndf/pdim
    inv_gamma_sqrt = math.sqrt(1/gamma)
    a = var*(1-inv_gamma_sqrt)**2
    b = var*(1+inv_gamma_sqrt)**2
    return a, b

def dmp(x, ndf, pdim, var=1, log=False):
    gamma = ndf/pdim
    a, b = mp_density(ndf, pdim, var)

    if not log:
        if gamma == 1 and x == 0 and 1/x > 0:
            d = math.inf
        elif x <= a and x >= b:
            d = 0
        else:
            d = gamma/(2*math.pi*var*x) * math.sqrt((x-a)*(b-x))
    else:
        if gamma == 1 and x == 0 and 1/x > 0:
            d = math.inf
        elif x <= a and x >= b:
            d = -math.inf
        else:
            d = (log(gamma) - (log(2) + log(math.pi) + log(var) + log(x)) +
                 0.5*log(x-a) + 0.5*log(b-x))
    return d

def pmp(q, ndf, pdim, var=1, lower_tail=True, log_p=False):
    gamma = ndf/pdim
    a, b = mp_density(ndf, pdim, var)
    f = lambda x: dmp(x, ndf, pdim, var)

    if lower_tail:
        if q <= a:
            p = 0
        elif q >= b:
            p = 1
        else:
            p = quad(f, a, q)[0]
        if gamma < 1 and q >= 0:
            p += (1 - gamma)
    else:
        if q <= a:
            p = min(1, gamma)
        elif q >= b:
            p = 0
        else:
            p = quad(f, q, b)[0]
        if gamma < 1 and q <= 0:
            p += (1 - gamma)

    return math.log(p) if log_p else p

class SplittableLinear(nn.Module):
    def __init__(self, in_features, out_features, alpha, beta, goodness_of_fit_cutoff,
                 name="splittable_linear", bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer1 = nn.Linear(in_features, out_features, bias=bias)
        self.layer2 = nn.Identity()
        self.splitted = False
        self.alpha = alpha
        self.beta = beta
        self.goodness_of_fit_cutoff = goodness_of_fit_cutoff
        self.name = name

        # Initialize weights
        nn.init.normal_(self.layer1.weight, mean=0, std=np.sqrt(1.0/in_features))
        if bias:
            nn.init.zeros_(self.layer1.bias)

    def forward(self, x):
        if self.splitted:
            x = self.layer2(self.layer1(x))
        else:
            x = self.layer1(x)
        return x

    @property
    def weight(self):
        return self.layer1.weight

    @property
    def bias(self):
        return self.layer1.bias

    @property
    def param_numbers(self):
        if self.splitted:
            return ((self.in_features + self.out_features) *
                   self.layer1.out_features)
        return self.in_features * self.out_features

    def get_matrix(self):
        matrix = self.layer1.weight.cpu().detach().numpy()
        if not self.splitted:
            return matrix
        matrix2 = self.layer2.weight.cpu().detach().numpy()
        return matrix2 @ matrix

    def fit_mp(self, U, singular_values, V, save_name, show):
        eigenvals = singular_values**2 / V.shape[0]
        eigenvals = np.sort(eigenvals)

        p = min(U.shape[0], V.shape[0])
        n = max(U.shape[0], V.shape[0])
        gamma = p / n

        # BEMA algorithm implementation
        significant_singulars = np.sum(singular_values > np.sqrt(n * self.alpha))

        # Use Tracy-Widom distribution
        tw1 = TracyWidom(beta=1)
        t_b = tw1.cdfinv(1-self.beta)
        lambda_plus = (1 + np.sqrt(gamma))**2 + t_b * n**(-2/3) * gamma**(-1/6) * (1 + np.sqrt(gamma))**(4/3)

        # Check goodness of fit
        error = self._compute_error(eigenvals, gamma)
        good_fit = error < self.goodness_of_fit_cutoff

        return np.sqrt(n * lambda_plus), good_fit

    def split(self, ratio, save_name, show=False):
        matrix = self.get_matrix()
        U, S, V = np.linalg.svd(matrix)
        Splus, good_fit = self.fit_mp(U, S, V, save_name, show=show)

        if not good_fit:
            return f" {self.name} no good fit"

        significant_singulars = np.sum(S > Splus)
        inner_dim = int((S.shape[0] - significant_singulars) * ratio) + significant_singulars

        if self.param_numbers <= (matrix.shape[0] + matrix.shape[1]) * inner_dim:
            if not self.splitted:
                new_weights = (U[:, :inner_dim] * S[None, :inner_dim]) @ V[:inner_dim, :]
                self.set_params("layer1", torch.from_numpy(new_weights).float(),
                            bias=None, change_bias=False)
            return f" {self.name} not enough param reduc"

        new_weights1 = np.sqrt(S)[:inner_dim, None] * V[:inner_dim, :]
        new_weights2 = U[:, :inner_dim] * np.sqrt(S)[None, :inner_dim]

        try:
            bias = nn.Parameter(self.layer1.bias.clone()) if self.layer1.bias is not None else None
        except AttributeError:
            bias = None

        # Get current device
        device = self.layer1.weight.device

        # Create new layers and immediately move them to the correct device
        layer1, layer2 = self._make_splitted_layers(inner_dim)
        self.layer1 = layer1.to(device)
        self.layer2 = layer2.to(device)

        # Convert numpy arrays to tensors and set parameters
        weights1 = torch.from_numpy(new_weights1).float()
        weights2 = torch.from_numpy(new_weights2).float()

        self.set_params("layer1", weights1, None)
        self.set_params("layer2", weights2, bias)

        self.splitted = True

        return f" {self.name} splitted, new dims {(self.in_features, inner_dim, self.out_features)}"

    def _make_splitted_layers(self, inner_dim):
        layer1 = nn.Linear(self.in_features, inner_dim, bias=False)
        layer2 = nn.Linear(inner_dim, self.out_features, bias=False)
        return layer1, layer2

    def set_params(self, which_layer, weight, bias, change_bias=True):
        assert which_layer in ["layer1", "layer2"]
        layer = getattr(self, which_layer)

        # Move new weights to the same device as the layer
        device = layer.weight.device
        weight = weight.to(device)

        layer.weight = nn.Parameter(weight)
        if change_bias:
            if bias is not None:
                bias = bias.to(device)  # Move bias to same device
                layer.bias = nn.Parameter(bias)
            else:
                layer.bias = None

    def _compute_error(self, eigenvals, gamma):
        # Implement error computation using L-infinity norm between theoretical CDF and empirical CDF
        # This is a simplified version - you might want to implement the full error computation
        return np.max(np.abs(np.sort(eigenvals) - np.linspace(0, gamma, len(eigenvals))))