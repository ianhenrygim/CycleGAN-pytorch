import torch
import torch.nn as nn
import itertools
from torch.optim import lr_scheduler
from .generator import ResnetGenerator
from .discriminator import PatchGANGDiscriminator

class CycleGAN():
    def __init__(self, opt):
        self.isTrain = True if opt.mode == "train" else False
        
        # pre-procssing
        self.n_residual_blocks = 9 if opt.size == 256 else 6

        # for printing
        self.loss_names = ['D_A_loss', 'D_B_loss', 'G_A_loss', 'G_B_loss', 'forward_cycle_loss', 'backward_cycle_loss']
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']

        # Generator
        self.G_A = ResnetGenerator(opt.input_nc, opt.output_nc, n_residual_blocks)
        self.G_B = ResnetGenerator(opt.output_nc, opt.input_nc, n_residual_blocks)

        # Discriminator
        self.D_A = PatchGANGDiscriminator(input_nc)
        self.D_B = PatchGANGDiscriminator(input_nc)

        # Loss
        # Todo : Add identity loss
        self.criterion_GAN = GANLoss().to("cuda")
        self.criterion_Cycle = torch.nn.L1Loss()
        
        # Todo : Image Buffer
        # -------------------

        # optimizer
        self.optimzer_G = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimzer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizers = [self.optimzer_G, self.optimzer_D]    # for scheduler

        # scheduler
        if self.isTrain:
            self.schedulers = [self.get_scheduler(optimizer) for optimizer in self.optimizers]

    def get_scheduler(optimizer):
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - 100) / float(100 + 1)
            return lr_l
        
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


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



