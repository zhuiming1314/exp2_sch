import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from torch.optim import lr_scheduler

class ReluConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None'):
        super(ReluConv2d, self).__init__()
        model = []

        model.append(nn.ReflectionPad2d(padding))
        model.append(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))
        model.append(nn.LeakyReLU(inplace=True))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



class ReluInsConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReluInsConv2d, self).__init__()
        model = []

        model.append(nn.ReflectionPad2d(padding))
        model.append(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0))
        model.append(nn.InstanceNorm2d(n_out, affine=False))
        model.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
        

class InsResBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=1, dropout=0):
        super(InsResBlock, self).__init__()
        model = []

        # 3*3 conv
        model.append(nn.ReflectionPad2d(padding=1))
        model.append(nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride))
        model.append(nn.InstanceNorm2d(n_out))
        model.append(nn.ReLU(inplace=True))
        
        n_in = n_out
        model.append(nn.ReflectionPad2d(padding=1))
        model.append(nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride))
        model.append(nn.InstanceNorm2d(n_out))

        if dropout > 0:
            model.append(nn.Dropout(p=dropout))
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        return self.model(x) + residual

class MisInsResBlock(nn.Module):
    def __init__(self, n_in, n_extra, stride=1, dropout=0):
        super(MisInsResBlock, self).__init__()
        
        n_out = n_in
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride),
            nn.InstanceNorm2d(n_out)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride),
            nn.InstanceNorm2d(n_out)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(n_in+n_extra, n_out+n_extra, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=False),
            nn.Conv2d(n_in+n_extra, n_out, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=False)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(n_in+n_extra, n_out+n_extra, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=False),
            nn.Conv2d(n_in+n_extra, n_out, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=False)
        )

        dpLayer = []
        if dropout > 0:
            dpLayer.append(nn.Dropout(p=dropout))
        
        self.dpLayer = nn.Sequential(*dpLayer)

    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))

        out1 = self.conv1(x)
        out2 = self.block1(torch.cat([out1, z_expand], dim=1))
        out3 = self.conv2(out2)
        out4 = self.block2(torch.cat([out3, z_expand], dim=1))
        out = out4 + residual
        return out

class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.ones(n_out, 1, 1))
        
    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return func.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
            return func.layer_norm(x, normalized_shape)

class ReluInsConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReluInsConvTranspose2d, self).__init__()

        model = []
        model.append(nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True))
        model.append(LayerNorm(n_out))
        model.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
