import torch
import torch.nn as nn

"""
    Let c7s1-k denote a 7×7 Convolution-InstanceNorm-ReLU layer with k ﬁlters and stride 1. 
    dk denotes a 3×3 Convolution-InstanceNorm-ReLU layer with k ﬁlters and stride 2. 
    Reﬂection padding was used to reduce artifacts. 
    Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same number of ﬁlters on both layer. 
    uk denotes a 3 × 3 fractional-strided-Convolution-InstanceNorm-ReLU layer with k ﬁlters and stride 1 2. 
    The network with 6 residual blocks consists of: c7s1-64,d128,d256,R256,R256,R256, R256,R256,R256,u128,u64,c7s1-3 
    The network with 9 residual blocks consists of: c7s1-64,d128,d256,R256,R256,R256, R256,R256,R256,R256,R256,R256,u128 u64,c7s1-3
"""
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks):
        super(ResnetGenerator, self).__init__()

        # c7s1-64
        model = [nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, 64, kernel_size=7),
                nn.InstanceNorm2d(64),
                nn.ReLU(True)]

        # d128 - downsampling
        model += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(True)]

        # d256 - downsampling
        model += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.ReLU(True)]

        # R256 * num of residual blocks(6 or 9)
        for _ in range(0, n_residual_blocks):
            model += [_ResidualBlock()]

        # u128 - upsampling
        model += [nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(True)]

        # u64 - upsampling
        model += [nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(True)]

        # c7s1-3
        model += [nn.ReflectionPad2d(3),
                nn.Conv2d(64, output_nc, kernel_size=7),
                nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class _ResidualBlock(nn.Module):
    def __init__(self, n_channel=256, kernel_size=3):
        super(_ResidualBlock, self).__init__()
        self.conv_block = self.build_conv_block(n_channel, kernel_size)
    
    def build_conv_block(self, n_channel, kernel_size):
        block = [nn.ReflectionPad2d(1), 
                nn.Conv2d(n_channel, n_channel, kernel_size, padding=0),
                nn.InstanceNorm2d(n_channel)]
        relu = nn.ReLU(True)

        return nn.Sequential(*block, relu, *block)

    def forward(self, x):
        out = x + self.conv_block(x)    # skip connection
        return out

# Test
if __name__ == "__main__":
    model = None
    model = ResnetGenerator(3, 3, 9)

    out_model = model(torch.randn(1, 3, 256, 256))
    print(out_model.shape)