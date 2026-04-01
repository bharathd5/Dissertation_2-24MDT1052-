import torch
from model import PrimaryTransformer

out = PrimaryTransformer(n_features=103)(torch.randn(4, 60, 103))
print('Output shape:', out.shape)
print('Model OK - ready to train')