import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()

def closs_entropy(target, logit):
    loss = 0
    for t, p in  zip(target, logit):
        loss += -t*torch.log(p+1e-10)
    return loss

def cross_entropy_loss(targets, predictions):

    loss = -torch.sum(targets * torch.log(predictions + 1e-10))

    return loss

target = torch.tensor([[0.2, 0.3, 0.1, 0, 0, 0, 0, 0.4],[0.2, 0.3, 0.1, 0, 0, 0, 0, 0.4],[0.2, 0.3, 0.1, 0, 0, 0, 0, 0.4]])
logit = torch.tensor([[0.1, 0.7, 0.05, 0, 0.02, 0.03, 0, 0.1],[0.1, 0.7, 0.05, 0, 0.02, 0.03, 0, 0.1],[0.1, 0.7, 0.05, 0, 0.02, 0.03, 0, 0.1]])

print(criterion(target, logit))
print(cross_entropy_loss(target, logit))