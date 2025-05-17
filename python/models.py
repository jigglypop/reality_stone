import torch
import torch.nn as nn
import reality_stone as rs

class LorentzMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, t=0.7):
        super().__init__()
        self.c = c
        self.t = t
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid+1, out_dim) * 0.01)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = x @ self.weights1 + self.bias1
        h = torch.tanh(h)
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        lorentz_h = rs.poincare_to_lorentz(h, self.c)
        lorentz_u = rs.poincare_to_lorentz(u, self.c)
        lorentz_z = rs.lorentz_layer(lorentz_h, lorentz_u, self.c, self.t)
        if torch.isnan(lorentz_z).any():
            lorentz_z = lorentz_h
        output = lorentz_z @ self.out_weights + self.out_bias
        return output

class KleinMLP(nn.Module):
    def __init__(self, in_dim=784, hid=128, out_dim=10, c=1e-3, t=0.7):
        super().__init__()
        self.c = c
        self.t = t
        self.weights1 = nn.Parameter(torch.randn(in_dim, hid) * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hid))
        self.weights2 = nn.Parameter(torch.randn(hid, hid) * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(hid))
        self.out_weights = nn.Parameter(torch.randn(hid, out_dim) * 0.01)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = x @ self.weights1 + self.bias1
        h = torch.tanh(h)
        u = h @ self.weights2 + self.bias2
        u = torch.sigmoid(u)
        klein_h = rs.poincare_to_klein(h, self.c)
        klein_u = rs.poincare_to_klein(u, self.c)
        klein_z = rs.klein_layer(klein_h, klein_u, self.c, self.t)
        if torch.isnan(klein_z).any():
            klein_z = klein_h
        output = klein_z @ self.out_weights + self.out_bias
        return output
