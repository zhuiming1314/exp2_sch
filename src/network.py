import functools
import torch
import torch.nn as nn
import net_module as net
from torch.optim import lr_scheduler

###############################################################
#-------------------------Encoders----------------------------#
###############################################################
class ContentEncoder(nn.Module):
    def __init__(self, input_dim_a, input_dim_b):
        super(ContentEncoder, self).__init__()

        # content encoder of domain A
        encA_c = []
        n_in = input_dim_a
        n_out = 64

        encA_c.append(net.ReluConv2d(n_in, n_out, kernel_size=7, stride=1, padding=3))
        
        for _ in range(1, 3):
            n_in = n_out
            n_out *= 2
            encA_c.append(net.ReluInsConv2d(n_in, n_out, kernel_size=3, stride=2, padding=1))

        n_in = n_out
        for _ in range(1, 4):
            encA_c.append(net.InsResBlock(n_in, n_out))
            
        # content encoder of domain B
        encB_c = []
        n_in = input_dim_b
        n_out = 64

        encB_c.append(net.ReluConv2d(n_in, n_out, kernel_size=7, stride=1, padding=3))

        for _ in range(1, 3):
            n_in = n_out
            n_out *= 2
            encB_c.append(net.ReluInsConv2d(n_in, n_out, kernel_size=3, stride=2, padding=1))

        n_in = n_out
        for _ in range(1, 4):
            encB_c.append(net.InsResBlock(n_in, n_out))

        self.encA_c = nn.Sequential(*encA_c)
        self.encB_c = nn.Sequential(*encB_c)
    
    def forward_a(self, xa):
        return self.encA_c(xa)

    def forward_b(self, xb):
        return self.encB_c(xb)    

    def forward(self, xa, xb):
        output_a = self.encA_c(xa)
        output_b = self.encB_c(xb)

        return output_a, output_b




class StyleEncoder(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, output_nc):
        super(StyleEncoder, self).__init__()

        # style encoder of domain a
        encA_s = []
        n_in = input_dim_a
        n_out = 64

        encA_s.append(net.ReluConv2d(n_in, n_out, kernel_size=7, stride=1, padding=3))

        for _ in range(1, 3):
            n_in = n_out
            n_out *= 2
            encA_s.append(net.ReluConv2d(n_in, n_out, kernel_size=4, stride=2, padding=1))

        n_in = n_out
        for _ in range(1, 3):
            encA_s.append(net.ReluConv2d(n_in, n_out, kernel_size=4, stride=2, padding=1))

        encA_s.append(nn.AdaptiveAvgPool2d(1))
        encA_s.append(nn.Conv2d(n_out, output_nc, kernel_size=1, stride=1, padding=0))

        # style encoder of domain b
        encB_s = []
        n_in = input_dim_b
        n_out = 64
        
        encB_s.append(net.ReluConv2d(n_in, n_out, kernel_size=7, stride=1, padding=3))

        for _ in range(1, 3):
            n_in = n_out
            n_out *= 2
            encB_s.append(net.ReluConv2d(n_in, n_out, kernel_size=4, stride=2, padding=1))

        n_in = n_out
        for _ in range(1, 3):
            encB_s.append(net.ReluConv2d(n_in, n_out, kernel_size=4, stride=2, padding=1))
        
        encB_s.append(nn.AdaptiveAvgPool2d(1))
        encB_s.append(nn.Conv2d(n_out, output_nc, kernel_size=1, stride=1, padding=0))

        self.encA_s = nn.Sequential(*encA_s)
        self.encB_s = nn.Sequential(*encB_s)

    def forward_a(self, xa):
        return self.encA_s(xa)

    def forward_b(self, xb):
        return self.encB_s(xb)

    def forward(self, xa, xb):
        output_a = self.encA_s(xa)
        output_b = self.encB_s(xb)

        return output_a, output_b
##############################################################
#-----------------Generators/Decoders------------------------#
##############################################################
class Generator(nn.Module):
    def __init__(self, output_dim_a, output_dim_b, nz):
        super(Generator, self).__init__()
        self.nz = nz

        # Generator of domain A
        n_in = 256
        n_out = n_in
        n_extra = n_in
        self.n_extra = n_extra
        
        self.decA_1 = net.MisInsResBlock(n_in, n_extra)
        self.decA_2 = net.MisInsResBlock(n_in, n_extra)
        self.decA_3 = net.MisInsResBlock(n_in, n_extra)
        self.decA_4 = net.MisInsResBlock(n_in, n_extra)

        decA_5 = []
        for _ in range(1, 3):
            n_in = n_out
            n_out = n_in // 2
            decA_5.append(net.ReluInsConvTranspose2d(n_in, n_out, kernel_size=3, stride=2, padding=1, output_padding=1))
        
        n_in = n_out
        decA_5.append(nn.ConvTranspose2d(n_in, output_dim_a, kernel_size=1, stride=1, padding=0))
        decA_5.append(nn.Tanh())

        self.decA_5 = nn.Sequential(*decA_5)

        self.mlpA = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_extra*4)
        )

        # Generator of domain B
        n_in = 256
        n_out = n_in
        n_extra = n_in
        
        self.decB_1 = net.MisInsResBlock(n_in, n_extra)
        self.decB_2 = net.MisInsResBlock(n_in, n_extra)
        self.decB_3 = net.MisInsResBlock(n_in, n_extra)
        self.decB_4 = net.MisInsResBlock(n_in, n_extra)

        decB_5 = []
        for _ in range(1, 3):
            n_in = n_out
            n_out = n_in // 2
            decB_5.append(net.ReluInsConvTranspose2d(n_in, n_out, kernel_size=3, stride=2, padding=1, output_padding=1))

        n_in = n_out
        decB_5.append(nn.ConvTranspose2d(n_in, output_dim_a, kernel_size=1, stride=1, padding=0))
        decB_5.append(nn.Tanh())

        self.decB_5 = nn.Sequential(*decB_5)

        self.mlpB = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_extra*4)
        )
    
    def forward_a(self, x, z):
        z = self.mlpA(z)
        z1, z2, z3, z4 = torch.split(z, self.n_extra, dim=1)
        z1, z2, z3, z4  = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()

        out1 = self.decA_1(x, z1)
        out2 = self.decA_2(out1, z2)
        out3 = self.decA_3(out2, z3)
        out4 = self.decA_4(out3, z4)
        out5 = self.decA_5(out4)

        return out5
    
    def forward_b(self, x, z):
        z = self.mlpB(z)
        z1, z2, z3, z4 = torch.split(z, self.n_extra, dim=1)
        z1, z2, z3, z4  = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()

        out1 = self.decB_1(x, z1)
        out2 = self.decB_2(out1, z2)
        out3 = self.decB_3(out2, z3)
        out4 = self.decB_4(out3, z4)
        out5 = self.decB_5(out4)

        return out5






#############################################################
#--------------------Discriminator--------------------------#
#############################################################
class Discriminator(nn.Module):
    def __init__(self, n_in, n_scale=3, n_layer=4, norm="None"):
        super(Discriminator, self).__init__()

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.Diss = nn.ModuleList()
        n_out = 64
        for _ in range(n_scale):
            self.Diss.append(self._make_net(n_in, n_out, n_layer, norm))
        
    def _make_net(self, n_in, n_out, n_layer, norm):
        model = []

        model.append(net.ReluConv2d(n_in, n_out, kernel_size=4, stride=2, padding=1, norm=norm))

        for _ in range(1, n_layer):
            n_in = n_out
            n_out *= 2
            model.append(net.ReluConv2d(n_in, n_out, kernel_size=4, stride=2, padding=1, norm=norm))
        model.append(nn.Conv2d(n_out, 1, kernel_size=1, stride=1, padding=0))

        return nn.Sequential(*model)

    def forward(self, x):
        outs = []
        for dis in self.Diss:
            outs.append(dis(x))
            x = self.downsample(x)

        return outs


###############################################################
#---------------------------Basic Functions-------------------#
###############################################################
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == "lambda":
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_schduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == "step":
        scheduler = lr_schduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        return NotImplementedError("no such learn rate policy")
    return scheduler