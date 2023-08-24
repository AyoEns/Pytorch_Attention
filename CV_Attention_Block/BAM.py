import numpy as np
import torch
from torch import nn
from torch.nn import init

class Flatten(nn.Module):
    '''打平'''
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Channel_Attention(nn.Module):
    '''通道注意力'''
    def __init__(self, channel, reduction=16, num_layers=3):
        super(Channel_Attention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        gate_channels = [channel]
        gate_channels += [channel // reduction] * num_layers
        gate_channels += [channel]

        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())
        for i in range(len(gate_channels) - 2):
            self.ca.add_module('fc%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.ca.add_module('bn%d' % i, nn.BatchNorm1d(gate_channels[i + 1])) # BN 1D
            self.ca.add_module("relu%d" % i, nn.ReLU())
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        #   1.全局平均池化
        res = self.avgpool(x)
        #   2.通过一系列FC + BN + relu层
        #   FC层的通道数由当前输入的num_layers决定，决定每一次的步长
        res = self.ca(res)
        res = res.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return res


class Spatial_Attention(nn.Module):
    '''空间注意力机制'''
    def __init__(self, channel, reduction=16, num_layers=3, dia_val=2):
        super(Spatial_Attention, self).__init__()
        self.sa = nn.Sequential()
        #  第一层为 1*1的卷积层Conv+BN+Relu 降维为channel // reduction
        self.sa.add_module('conv_reduce1',
                           nn.Conv2d(kernel_size=1, in_channels=channel, out_channels=channel // reduction))
        self.sa.add_module('bn_reduce1', nn.BatchNorm2d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU())
        #  中间层堆叠 3 * 3 的卷积Conv+BN+Relu 通道不变
        for i in range(num_layers):
            self.sa.add_module('conv_%d' % i, nn.Conv2d(kernel_size=3, in_channels=channel // reduction,
                                                        out_channels=channel // reduction, padding=1, dilation=dia_val))
            self.sa.add_module('bn_%d' % i, nn.BatchNorm2d(channel // reduction))
            self.sa.add_module('relu_%d' % i, nn.ReLU())
        #   最后一层 压缩为通道数为1的卷积层 1*h*w的卷积层
        self.sa.add_module('last_conv', nn.Conv2d(channel // reduction, 1, kernel_size=1))

    def forward(self, x):
        res = self.sa(x)
        #   返回与x大小相同的权重
        res = res.expand_as(x)
        return res


class BAM_Block(nn.Module):
    def __init__(self, c1, reduction=16, dia_val=2):
        super(BAM_Block, self).__init__()
        self.ca = Channel_Attention(channel=c1, reduction=reduction)
        self.sa = Spatial_Attention(channel=c1, reduction=reduction, dia_val=dia_val)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        #   空间注意力
        sa_out = self.sa(x)
        #   通道注意力
        ca_out = self.ca(x)
        #   权重相加
        weight = self.sigmoid(sa_out + ca_out)
        #   特征加强
        out = (1 + weight) * x
        return out

if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    bam = BAM_Block(c1=512, reduction=16, dia_val=2)
    output = bam(input)
    print(output.shape)