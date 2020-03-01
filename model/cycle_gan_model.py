import torch
import torch.nn as nn
from torch.nn import init
import itertools
from torch.optim import lr_scheduler
from .generator import ResnetGenerator
from .discriminator import PatchGANGDiscriminator
from .image_buffer import ImageBuffer

class CycleGAN():
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = True if opt.mode == "train" else False
        self.useIdentity = True if opt.identity_loss else False
        
        # pre-procssing
        self.n_residual_blocks = 9 if opt.img_size == 256 else 6

        # for printing
        self.loss_names = ['D_A_loss', 'D_B_loss', 'G_A_loss', 'G_B_loss', 'forward_cycle_loss', 'backward_cycle_loss', 'idt_A_loss', 'idt_B_loss']
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']

        # Generator
        self.G_A = ResnetGenerator(opt.input_nc, opt.output_nc, self.n_residual_blocks)
        self.G_B = ResnetGenerator(opt.output_nc, opt.input_nc, self.n_residual_blocks)

        # Discriminator
        self.D_A = PatchGANGDiscriminator(opt.input_nc)
        self.D_B = PatchGANGDiscriminator(opt.input_nc)

        # Loss
        # Todo : Add identity loss
        self.criterion_GAN = GANLoss().to("cuda")
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_Ientity = torch.nn.L1Loss()

        self.idt_A_loss = 0
        self.idt_B_loss = 0
        
        # Image Buffer
        self.fake_A_buffer = ImageBuffer(opt.img_buffer_size)
        self.fake_B_buffer = ImageBuffer(opt.img_buffer_size)

        # optimizer
        self.optimzer_G = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimzer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizers = [self.optimzer_G, self.optimzer_D]    # for scheduler

        # scheduler
        if self.isTrain:
            self.schedulers = [self.get_scheduler(optimizer) for optimizer in self.optimizers]

    def setup(self):
        def weights_init(m): 
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                init.normal_(m.weight.data, 0.0, 0.2)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.G_A.to("cuda")
        self.G_B.to("cuda")
        self.D_A.to("cuda")
        self.D_B.to("cuda")

        self.G_A.apply(weights_init)
        self.G_B.apply(weights_init)
        self.D_A.apply(weights_init)
        self.D_B.apply(weights_init)

    def forward(self, input):
        self.real_A = input['A'].to("cuda")
        self.real_B = input['B'].to("cuda")

        self.fake_B = self.G_A(self.real_A)
        self.regen_A = self.G_B(self.fake_B)
        self.fake_A = self.G_A(self.real_B)
        self.regen_B = self.G_B(self.fake_A)

    def backward_G(self):
        # GAN Loss
        self.G_A_loss = self.criterion_GAN(self.D_A(self.fake_B), True)
        self.G_B_loss = self.criterion_GAN(self.D_B(self.fake_A), True)
        if (self.opt.identity_loss):    # identity loss
            self.idt_A = self.G_A(self.real_B)
            self.idt_A_loss = self.criterion_Ientity(self.idt_A, self.real_B) * 10.0 * 0.5
            self.idt_B = self.G_B(self.real_A)
            self.idt_B_loss = self.criterion_Ientity(self.idt_B, self.real_A) * 10.0 * 0.5

        # Cycle Loss
        self.forward_cycle_loss = self.criterion_cycle(self.regen_A, self.real_A) * 10.0 #lambda
        self.backward_cycle_loss = self.criterion_cycle(self.regen_B, self.real_B) * 10.0 #lambda

        # full objective
        self.full_loss = self.G_A_loss + self.G_B_loss + self.forward_cycle_loss + self.backward_cycle_loss + self.idt_A_loss + self.idt_B_loss

        # calculate gradients
        self.full_loss.backward()

    def backward_D(self):
        # D_A
        fake_B = self.fake_B_buffer.query(self.fake_B)
        D_A_loss_real = self.criterion_GAN(self.D_A(self.real_B), True)
        D_A_loss_fake = self.criterion_GAN(self.D_A(fake_B.detach()), False)
        D_A_loss = D_A_loss_real * 0.5 + D_A_loss_fake * 0.5
        D_A_loss.backward()
        self.D_A_loss = D_A_loss

        # D_B
        fake_A = self.fake_A_buffer.query(self.fake_A)
        D_B_loss_real = self.criterion_GAN(self.D_B(self.real_A), True)
        D_B_loss_fake = self.criterion_GAN(self.D_B(fake_A.detach()), False)
        D_B_loss = D_B_loss_real * 0.5 + D_B_loss_fake * 0.5
        D_B_loss.backward()
        self.D_B_loss = D_B_loss

    def get_scheduler(self, optimizer):
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - 100) / float(100 + 1)
            return lr_l
        
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    def set_requires_grad(self, netType="D"):
        nets = [self.D_A, self.D_B]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = True



class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.loss = nn.MSELoss()
    
    def __call__(self, pred, isRealTarget):
        return self.loss(pred, self.get_target_tensor(pred, isRealTarget))

    def get_target_tensor(self, pred, isRealTarget):
        if isRealTarget:
            return self.real_label.expand_as(pred)
        else:
            return self.fake_label.expand_as(pred)



