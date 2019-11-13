import torch.nn as nn
from models.resnet import BasicBlock as ResNetBlock


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, ngf=64, nb=8):
        super(Generator, self).__init__()
        self.down_convs = nn.Sequential(
            # k7n64s1
            nn.Conv2d(in_channels, ngf, 7, 1, 3),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            # k3n128s2
            nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
            # k3n128s1
            nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            # k3n256s2
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
            # k3n256s1
            nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU()
        )

        self.resnet_blocks = []
        for i in range(nb):
            # k3n256s1
            self.resnet_blocks.append(ResNetBlock(ngf * 4, ngf * 4, 3, 1, 1))
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.up_convs = nn.Sequential(
            # k3n128s1/2
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1),
            # k3n128s1
            nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 1, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            # k3n64s1/2
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1),
            # k3n64s1
            nn.ConvTranspose2d(ngf, ngf, 3, 1, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            # k7n3s1
            nn.Conv2d(ngf, out_channels, 7, 1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.down_convs(x)
        x = self.resnet_blocks(x)
        x = self.up_convs(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, ndf=32):
        super(Discriminator, self).__init__()
        self.convs = nn.Sequential(
            # k3n32s1
            nn.Conv2d(in_channels, ndf, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # k3n64s2
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1),
            nn.LeakyReLU(0.2),
            # k3n128s1
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            # k3n128s2
            nn.Conv2d(ndf * 4, ndf * 4, 3, 2, 1),
            nn.LeakyReLU(0.2),
            # k3n256s1
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
            # k3n256s1
            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
            # k3n1s1
            nn.Conv2d(ndf * 8, out_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.convs(x)
        return x
