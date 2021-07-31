import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit


class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MyConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return self.conv(x)


class MemoryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MemoryConv2d, self).__init__()
        self.Wz = MyConv2d(in_channels, out_channels, kernel_size)
        self.Wi = MyConv2d(in_channels, out_channels, kernel_size)
        self.Wf = MyConv2d(in_channels, out_channels, kernel_size)
        self.Wo = MyConv2d(in_channels, out_channels, kernel_size)
        self.c = None

    def reset(self, shape, device):
        self.c = torch.zeros(shape).to(device)

    def forward(self, x):
        z = self.Wz(x)
        z = torch.tanh(z)

        i = self.Wi(x)
        i = torch.sigmoid(i)

        f = self.Wf(x)
        f = torch.sigmoid(f)

        self.c = i * z + f * self.c

        o = self.Wo(x)
        o = torch.sigmoid(o)

        x = o * torch.tanh(self.c)

        return x


class Conv2DwithLayerNorm(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size):
        super(Conv2DwithLayerNorm, self).__init__()
        self.conv_ = nn.Conv2d(input_ch, output_ch, kernel_size, bias=False, padding=kernel_size // 2)
        self.norm_ = nn.LayerNorm((output_ch, 28, 28))

    def forward(self, x):
        t = self.conv_.forward(x)
        t = self.norm_.forward(t)
        return t


class MemoryConv2DwithLayerNorm(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size):
        super(MemoryConv2DwithLayerNorm, self).__init__()
        self.conv_ = MemoryConv2d(input_ch, output_ch, kernel_size)
        self.norm_ = nn.LayerNorm((output_ch, 28, 28))

    def reset(self, shape, device):
        self.conv_.reset(shape, device)

    def forward(self, x):
        t = self.conv_.forward(x)
        t = self.norm_.forward(t)
        return t


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, kernel_size, reduction):
        super(ResidualBlock, self).__init__()
        self.conv_and_norm0_ = MemoryConv2DwithLayerNorm(channel_num, channel_num, kernel_size)
        self.conv_and_norm1_ = MemoryConv2DwithLayerNorm(channel_num, channel_num, kernel_size)
        self.linear0_ = nn.Linear(channel_num, channel_num // reduction, bias=False)
        self.linear1_ = nn.Linear(channel_num // reduction, channel_num, bias=False)

    def reset(self, shape, device):
        self.conv_and_norm0_.reset(shape, device)
        self.conv_and_norm1_.reset(shape, device)

    def forward(self, x):
        t = x
        t = self.conv_and_norm0_.forward(t)
        t = F.relu(t)
        t = self.conv_and_norm1_.forward(t)

        y = F.avg_pool2d(t, [t.shape[2], t.shape[3]])
        y = y.view([-1, t.shape[1]])
        y = self.linear0_.forward(y)
        y = F.relu(y)
        y = self.linear1_.forward(y)
        y = torch.sigmoid(y)
        y = y.view([-1, t.shape[1], 1, 1])
        t = t * y

        t = F.relu(x + t)
        return t


class MemoryResNet(nn.Module):
    def __init__(self, input_channel_num, channel_num, n_classes, itr_num, kernel_size=3, reduction=8):
        super(MemoryResNet, self).__init__()
        self.first_conv_and_norm_ = Conv2DwithLayerNorm(input_channel_num, channel_num, 3)
        self.block = ResidualBlock(channel_num, kernel_size, reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channel_num, n_classes)
        self.itr_num = itr_num
        self.channel_num = channel_num

    def forward(self, x):
        self.block.reset((x.shape[0], self.channel_num, x.shape[2], x.shape[3]), x.device)
        x = self.first_conv_and_norm_.forward(x)
        x = F.relu(x)
        for _ in range(self.itr_num):
            x = self.block.forward(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
