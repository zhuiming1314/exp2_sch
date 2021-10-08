import torch
import torch.nn as nn
import torch.nn.functional as F

class CalcRecLoss():
    def __init__(self):
        self.loss_func = nn.L1Loss()

    def __call__(self, origin, rec):
        return self.loss_func(origin, rec)

class CalcGANLoss():
    '''pred is tensor type and target is bool type'''
    def __init__(self, gpu):
        self.loss_func = nn.MSELoss()
        self.gpu = gpu

    def __call__(self, pred, target):
        if target:
            target_tensor = torch.ones_like(pred).cuda(self.gpu)
        else:
            target_tensor = torch.zeros_like(pred).cuda(self.gpu)

        return self.loss_func(pred, target_tensor)
