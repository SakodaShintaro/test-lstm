import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit


class Conv2dWithLayerNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Conv2dWithLayerNorm, self).__init__()
        self.conv_ = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm_ = nn.LayerNorm((out_channels, 28, 28))

    def forward(self, x):
        x = self.conv_(x)
        x = self.norm_(x)
        return x


class Branch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Branch, self).__init__()
        hidden_channels = in_channels // 2
        self.conv0_ = Conv2dWithLayerNorm(in_channels, hidden_channels, kernel_size=kernel_size)
        self.conv1_ = Conv2dWithLayerNorm(hidden_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv0_(x)
        x = torch.relu(x)
        x = self.conv1_(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, hidden_channels, memory_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        sum_channels = hidden_channels + memory_channels
        self.conv_input_gate_ = Branch(sum_channels, memory_channels, kernel_size)
        self.conv_forget_gate_ = Branch(sum_channels, memory_channels, kernel_size)
        self.conv_update_value_ = Branch(sum_channels, memory_channels, kernel_size)
        self.conv_forward_ = Branch(sum_channels, hidden_channels, kernel_size)

    def forward(self, x, memory):
        cat = torch.cat([x, memory], dim=1)

        input_gate = torch.sigmoid(self.conv_input_gate_(cat))
        forget_gate = torch.sigmoid(self.conv_forget_gate_(cat))
        update_value = torch.tanh(self.conv_update_value_(cat))
        next_x = torch.relu(x + self.conv_forward_(cat))

        memory = input_gate * update_value + forget_gate * memory

        return next_x, memory


class MemoryResNet2(nn.Module):
    def __init__(self, input_channel_num, hidden_channel_num, n_classes, itr_num, kernel_size=3, reduction=8):
        super(MemoryResNet2, self).__init__()
        self.hidden_channel_num = hidden_channel_num - 1
        self.memory_channel_num = hidden_channel_num - self.hidden_channel_num

        self.first_conv_ = Conv2dWithLayerNorm(input_channel_num, self.hidden_channel_num, 3)
        self.block = ResidualBlock(self.hidden_channel_num, self.memory_channel_num, kernel_size)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.hidden_channel_num, n_classes)
        self.itr_num = itr_num

    def forward(self, x):
        x = self.first_conv_.forward(x)
        x = F.relu(x)

        memory = torch.zeros((x.shape[0], self.memory_channel_num, x.shape[2], x.shape[3])).to(x.device)
        for _ in range(self.itr_num):
            x, memory = self.block.forward(x, memory)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
