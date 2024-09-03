# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch

from torch import nn
from torch.nn import functional
import torch.utils
import torch.distributions
import torchvision
import lightning.pytorch as pl


import numpy as np
import matplotlib.pyplot as plt

# If you don't have access to a GPU use device='cpu'
device = "cuda"
