import torch
import torch.nn as nn

"""
 For discriminator networks, we use 70 × 70 PatchGAN [22]. 
 Let Ck denote a 4×4 Convolution-InstanceNorm-LeakyReLU layer with k ﬁlters and stride 2. 
 After the last layer, we apply a convolution to produce a 1-dimensional output.
 We do not use InstanceNorm for the ﬁrst C64 layer. We use leaky ReLUs with a slope of 0.2. 
 The discriminator architecture is: C64-C128-C256-C512
"""
class PatchGANGDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super(PatchGANGDiscriminator, self).__init__()

        # C64 -first layer (don't use InstanceNorm)
        model = [nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(.2, True)]

        # C128
        model += [nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, True)]

        # C256
        model += [nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, True)]

        # C512
        model += [nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, True)]

        # last layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)    # shape -> [1,1,15,15]

# Test
if __name__ == "__main__":
    model = None
    model = PatchGANGDiscriminator(3)

    x = torch.randn(1, 3, 256, 256)

    out_model = model(x)