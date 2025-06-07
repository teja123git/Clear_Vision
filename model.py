import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels + growth_rate * i, growth_rate, 3, padding=1)
            for i in range(num_layers)
        ])
        out_channels = in_channels + growth_rate * num_layers
        self.ca = ChannelAttention(out_channels)
        self.reduce = nn.Conv2d(out_channels, in_channels, 1)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = torch.cat([out, layer(out)], dim=1)
        ca_weights = self.ca(out)
        attended_out = out * ca_weights
        reduced_out = self.reduce(attended_out)
        return reduced_out + x

class Generator(nn.Module):
    def __init__(self, num_rrdb=6):  
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.rrdb = nn.Sequential(*[RDB(64, 32, 2) for _ in range(num_rrdb)])
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        self.conv3 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        initial = self.conv1(x)
        x = self.rrdb(initial)
        x = self.conv2(x) + initial
        x = self.upsample(x)
        return torch.tanh(self.conv3(x))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 64, 3, 2, 1)), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1)), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(128, 128, 3, 2, 1)), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1)), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 2, 1)), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1)), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(512, 512, 3, 2, 1)), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)