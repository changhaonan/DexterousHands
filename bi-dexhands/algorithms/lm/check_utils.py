""" Utils functions for checking
"""

import torch

def X(x):
    return x[:, 0]

def Y(x):
    return x[:, 1]

def Z(x):
    return x[:, 2]

def DIST(x, y):
    return torch.norm(x - y, dim=1)

def NORM(x):
    return torch.norm(x, dim=1)