from typing import ForwardRef
import network
import torch
import torch.nn as nn
import criterion

class TwinsNet(nn.Module):
    def __init__(self, opts):
        super(TwinsNet, self).__init__()
        
        # parameters
        lr = 0.0001
        betas = (0.5, 0.999)
        weight_decay = 0.0001
        self.batch_size = opts.batch_size
        self.nz = 8
        self.net_list = []
        self.opt_list = []
        self.dis_opt_list = []
        self.net_name = ["encA_c", "encB_c", "encA_s", "encB_s", "genA", "genB", "disA", "disB"]
        self.opt_name = ["encA_c_opt", "encB_c_opt", "encA_s_opt", "encB_s_opt", "genA_opt", "genB_opt"]
        self.dis_opt_name = ["disA_opt", "disB_opt"]
        # content encoders
        self.encA_c = network.ContentEncoder(opts.input_dim_a)
        self.encB_c = network.ContentEncoder(opts.input_dim_b)
        self.net_list.append(self.encA_c)
        self.net_list.append(self.encB_c)

        # style encoders
        self.encA_s = network.StyleEncoder(opts.input_dim_a, self.nz)
        self.encB_s = network.StyleEncoder(opts.input_dim_b, self.nz)
        self.net_list.append(self.encA_s)
        self.net_list.append(self.encB_s)

        # generator
        self.genA = network.Generator(opts.input_dim_a, nz=self.nz)
        self.genB = network.Generator(opts.input_dim_b, nz=self.nz)
        self.net_list.append(self.genA)
        self.net_list.append(self.genB)

        # discirminators
        self.disA = network.Discriminator(opts.input_dim_a, opts.dis_scale)
        self.disB = network.Discriminator(opts.input_dim_a, opts.dis_scale)
        self.net_list.append(self.disA)
        self.net_list.append(self.disB)

        # loss functions
        self.calc_rec_loss = criterion.CalcRecLoss()
        self.calc_gan_loss = criterion.CalcGANLoss(opts.gpu)

        # optimizers
        self.encA_c_opt = torch.optim.Adam(self.encA_c.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.encB_c_opt = torch.optim.Adam(self.encB_c.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.opt_list.append(self.encA_c_opt)
        self.opt_list.append(self.encB_c_opt)

        self.encA_s_opt = torch.optim.Adam(self.encA_s.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.encB_s_opt = torch.optim.Adam(self.encB_s.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.opt_list.append(self.encA_s_opt)
        self.opt_list.append(self.encB_s_opt)

        self.genA_opt = torch.optim.Adam(self.genA.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.genB_opt = torch.optim.Adam(self.genB.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.opt_list.append(self.genA_opt)
        self.opt_list.append(self.genB_opt)

        self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.dis_opt_list.append(self.disA_opt)
        self.dis_opt_list.append(self.disB_opt)

    def set_gpu(self, gpu):
        self.gpu = gpu
        for net in self.net_list:
            net.cuda(self.gpu)
        '''
        self.enc_c.cuda(self.gpu)
        self.enc_s.cuda(self.gpu)
        self.gen.cuda(self.gpu)
        self.disA.cuda(self.gpu)
        self.disB.cuda(self.gpu)
        '''
    def initialize(self):
        for net in self.net_list:
            net.apply(network.gaussian_weights_init)
        '''
        self.enc_c.apply(network.gaussian_weights_init)
        self.enc_s.apply(network.gaussian_weights_init)
        self.gen.apply(network.gaussian_weights_init)
        self.disA.apply(network.gaussian_weights_init)
        self.disB.apply(network.gaussian_weights_init)
        '''
    def set_scheduler(self, opts, last_ep=0):
        for opt in self.opt_list:
            network.get_scheduler(opt, opts, last_ep)
        for opt in self.dis_opt_list:
            network.get_scheduler(opt, opts, last_ep)
        '''
        self.enc_c_sch = network.get_scheduler(self.enc_c_opt, opts, last_ep)
        self.enc_s_sch = network.get_scheduler(self.enc_s_opt, opts, last_ep)
        self.gen_sch = network.get_scheduler(self.gen_opt, opts, last_ep)
        self.disA_sch = network.get_scheduler(self.gan_opt, opts, last_ep)
        self.disB_sch = network.get_scheduler(self.disB_opt, opts, last_ep)
        '''

    def forward(self):
        # get real content encode
        self.real_content_a = self.encA_c.forward(self.input_a)
        self.real_content_b = self.encB_c.forward(self.input_b)

        # get real style encode
        self.real_style_a = self.encA_s.forward(self.input_a)
        self.real_style_b = self.encB_s.forward(self.input_b)

        # generate imgae of content b and style a
        self.output_fake_a = self.genA.forward(self.real_content_b, self.real_style_b)
        self.output_fake_b = self.genB.forward(self.real_content_a, self.real_style_b)

        # get content encode form b and style encode form a
        self.fake_content_b = self.encB_c.forward(self.output_fake_a)
        self.fake_style_a = self.encA_s.forward(self.output_fake_a)
        # get content encode from a and style encode from b
        self.fake_content_a = self.encA_c.forward(self.output_fake_b)
        self.fake_style_b = self.encB_s.forward(self.output_fake_b)

        # generate image of real content a and fake style a
        self.rec_a1 = self.genA.forward(self.fake_content_a, self.real_style_a)
        self.rec_a2 = self.genA.forward(self.real_content_a, self.fake_style_a)

        # generate image of fake b content and real style b
        self.rec_b1 = self.genB.forward(self.fake_content_b, self.real_style_b)
        self.rec_b2= self.genB.forward(self.real_content_b, self.fake_style_b)

        self.image_display = torch.cat((self.input_a[0:1].detach().cpu(), self.input_b[0:1].detach().cpu(),
                                        self.rec_a1[0:1].detach().cpu(), self.rec_a2[0:1].detach().cpu(),
                                        self.rec_b1[0:1].detach().cpu(), self.rec_b2[0:1].detach().cpu(),
                                        self.output_fake_a[0:1].detach().cpu(), self.output_fake_b[0:1].detach().cpu()))
    def backward_gen(self):
        loss_gen_a = 0

        outs_fake_a = self.disA.forward(self.output_fake_a)
        for out in outs_fake_a:
            loss_gen_a += self.calc_gan_loss(out, True)
        
        loss_gen_b = 0
        outs_fake_b = self.disB.forward(self.output_fake_b)
        for out in outs_fake_b:
            loss_gen_b += self.calc_gan_loss(out, True)

        '''
        for out in outs_fake:
            o = nn.functional.sigmoid(out)
            all_ones = torch.ones_like(o).cuda(self.gpu)
            loss_gen += nn.functional.binary_cross_entropy(o, all_ones)
        '''
        print("loss_gen_a: {}, loss_gen_b:{}".format(loss_gen_a, loss_gen_b))
        return loss_gen_a, loss_gen_b

    def backward_rec(self):
        loss_rec_a = self.calc_rec_loss(self.input_a, self.rec_a1) +\
                        self.calc_rec_loss(self.input_a, self.rec_a2)
        loss_rec_b = self.calc_rec_loss(self.input_b, self.rec_b1) +\
                        self.calc_rec_loss(self.input_b, self.rec_b2)
        #loss_rec_content = torch.nn.L1Loss()(self.real_content_a, self.fake_content_a)
        #loss_rec_style = torch.nn.L1Loss()(self.real_style_b, self.fake_style_b)

        print("loss_rec_a: {}, loss_rec_b: {}".format(loss_rec_a, loss_rec_b))
        return loss_rec_a, loss_rec_b


    def update_enc_gen(self):
        '''
            # update encA_c, encA_s, genA
            self.encA_c_opt.zero_grad()
            self.encA_s_opt.zero_grad()
            self.genA_opt.zero_grad()
        '''
        for opt in self.opt_list:
            opt.zero_grad()

        loss_gen_a, loss_gen_b = self.backward_gen()
        loss_rec_a, loss_rec_b = self.backward_rec()
        loss_a = loss_gen_a + loss_rec_a
        loss_b = loss_gen_b + loss_rec_b

        loss_a.backward(retain_graph=True)
        loss_b.backward(retain_graph=True)

        for opt in self.opt_list:
            opt.step()

        '''
            self.encA_c_opt.step()
            self.encA_s_opt.step()
            self.genA_opt.step()
            self.encB_c_opt.step()
            self.encB_s_opt.step()
            self.genB_opt.step()
        '''
    def backward_dis(self):
        pred_fake_a = self.disA.forward(self.output_fake_a)
        pred_real_a = self.disA.forward(self.input_a)

        pred_fake_b = self.disB.forward(self.output_fake_b)
        pred_real_b = self.disB.forward(self.input_b)

        loss_dis_a = 0
        loss_dis_b = 0

        for _, (fake_a, real_a) in enumerate(zip(pred_fake_a, pred_real_a)):
            loss_dis_fake = self.calc_gan_loss(fake_a, False)
            loss_dis_real = self.calc_gan_loss(real_a, True)
            loss_dis_a += 0.5 * (loss_dis_fake + loss_dis_real)

        for _, (fake_b, real_b) in enumerate(zip(pred_fake_b, pred_real_b)):
            loss_dis_fake = self.calc_gan_loss(fake_b, False)
            loss_dis_real = self.calc_gan_loss(real_b, True)
            loss_dis_b += 0.5 * (loss_dis_fake + loss_dis_real)



        '''
        for _, (fake_a, real_a) in enumerate(zip(pred_fake_a, pred_real_a)):
            out_fake = nn.functional.sigmoid(fake)
            out_real = nn.functional.sigmoid(real)
            all_zeros = torch.zeros_like(out_fake).cuda(self.gpu)
            all_ones = torch.ones_like(out_real).cuda(self.gpu)
            loss_dis_fake = nn.functional.binary_cross_entropy(out_fake, all_zeros)
            loss_dis_real = nn.functional.binary_cross_entropy(out_real, all_ones)

            loss_dis += loss_dis_fake + loss_dis_real
        '''
        return loss_dis_a, loss_dis_b

    def update_dis(self, input_a, input_b):
        self.input_a = input_a
        self.input_b = input_b

        # encode and generate first
        self.forward()

        # update dis
        '''
            self.disB_opt.zero_grad()
            loss_dis = self.backward_dis()
            loss_dis.backward(retain_graph=True)
            self.disB_opt.step()
        '''
        for opt in self.dis_opt_list:
            opt.zero_grad()

        loss_dis_a, loss_dis_b = self.backward_dis()

        loss_dis_a.backward(retain_graph=True)
        loss_dis_b.backward(retain_graph=True)

        for opt in self.dis_opt_list:
            opt.zero_grad()

    def save_model(self, filename, ep, total_iter):
        '''
        state = {
            "enc_c": self.enc_c.state_dict(),
            "enc_s": self.enc_s.state_dict(),
            "gen": self.gen.state_dict(),
            "disB": self.disB.state_dict(),
            "enc_c_opt": self.enc_c_opt.state_dict(),
            "enc_s_opt": self.enc_s_opt.state_dict(),
            "gen_opt": self.gen_opt.state_dict(),
            "disB_opt": self.disB_opt.state_dict(),
            "ep": ep,
            "total_iter": total_iter
        }
        '''
        state = {}
        for _, (name, net) in enumerate(zip(self.net_name, self.net_list)):
            state[name] = net.state_dict()
        for _, (name, net) in enumerate(zip(self.opt_name, self.opt_list)):
            state[name] = net.state_dict()
        for _, (name, net) in enumerate(zip(self.dis_opt_name, self.dis_opt_list)):
            state[name] = net.state_dict()

        torch.save(state, filename)


    def resume(self, filename, train=True):
        checkpoint = torch.load(filename)
        
        if train:
            for _, (name, net) in enumerate(zip(self.net_name, self.net_list)):
                net.load_state_dict(checkpoint[name])
            for _, (name, net) in enumerate(zip(self.opt_name, self.opt_list)):
                net.load_state_dict(checkpoint[name])
            for _, (name, net) in enumerate(zip(self.dis_opt_name), self.dis_opt_list):
                net.load_state_dict(checkpoint[name])

            return checkpoint["ep"], checkpoint["total_iter"]
        '''
            self.enc_c.load_state_dict(checkpoint["enc_c"])
            self.enc_s.load_state_dict(checkpoint["enc_s"])
            self.gen.load_state_dict(checkpoint["gen"])
            self.disB.load_state_dict(checkpoint["disB"])
            self.enc_c_opt.load_state_dict(checkpoint["enc_c_opt"])
            self.enc_s_opt.load_state_dict(checkpoint["enc_s_opt"])
            self.gen_opt.load_state_dict(checkpoint["gen_opt"])
            self.disB_opt.load_state_dict(checkpoint["disB_opt"])
        '''
    
    def assemble_outputs(self):
        img_a = self.input_a.detach()
        img_b = self.input_b.detach()
        img_b_to_a = self.output_fake_b.detach
        img_a_to_b = self.output_fake_b.detach()
        rec_a1 = self.rec_a1.detach()
        rec_a2 = self.rec_a2.detach()
        rec_b1 = self.rec_b1.detach()
        rec_b2 = self.rec_b2.detach()

        row = torch.cat((img_a[0:1, ::], img_b[0:1, ::], rec_a1[0:1, ::], rec_a2[0:1, ::], rec_b1[0:1, ::], rec_b2[0:1, ::],
                            img_a_to_b[0:1, ::], img_b_to_a[0:1, ::]), 3)
        if self.batch_size >= 2:
            row2 = torch.cat((img_a[1:2, ::], img_b[1:2, ::], rec_a1[1:2, ::], rec_a2[1:2, ::], rec_b1[1:2, ::], rec_b2[1:2, ::],
                            img_a_to_b[1:2, ::], img_b_to_a[1:2, ::]), 3)
            row = torch.cat((row, row2), 2)
        return row

    def forward_transfer(self, input_a, input_b):
        content_a, content_b = self.enc_c(input_a, input_b)
        style_a, style_b = self.enc_s(input_a, input_b)

        output = self.gen.forward_b(content_a, style_b)

        return output




