import torch
from torch import nn
from torch.nn import init


class ChannelAttneitn(nn.Module):
    #   通道注意力机制
    '''
    通道注意力机制，通道注意力机制的实现可以分为两个部分，
    我们会对输入进来的单个特征层，分别进行全局平均池化和全局最大池化。
    之后对平均池化和最大池化的结果，利用共享的全连接层进行处理，
    对处理后的两个结果进行相加，然后取一个sigmoid，
    此时获得了输入特征层每一个通道的权值（0-1之间）。
    '''
    def __init__(self, c1, redution=16):
        super(ChannelAttneitn, self).__init__()
        #   通道注意力机制包含2种池化
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #   分别通过两层1X1卷积
        self.stage = nn.Sequential(
            nn.Conv2d(c1, c1 // redution, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(c1 // redution, c1, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.act_func = nn.Sigmoid()

    def forward(self, x):
        weight_max = self.maxpool(x)
        weight_avg = self.avgpool(x)
        weight_max = self.stage(weight_max)
        weight_avg = self.stage(weight_avg)
        weight = self.act_func(weight_max + weight_avg)     #    注意这里是加
        return weight

class SpatialAttention(nn.Module):
    #   空间注意力机制
    '''
    空间注意力机制，我们会对输入进来的特征层，
    在每一个特征点的通道上取最大值和平均值。
    之后将这两个结果进行一个堆叠,
    利用一次通道数为1的卷积调整通道数然后取一个sigmoid，
    此时我们获得了输入特征层每一个特征点的权值（0-1之间）
    '''
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernal size must be 3 or 7"
        padding = 3 if kernel_size== 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding, bias=False)
        self.act_func = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        weight = self.act_func(x)
        return weight


class CBAM_Attention(nn.Module):
    #   CBAM通道和位置注意力机制
    '''
    先进行通道注意力加强、后进行空间注意力加强
    '''
    def __init__(self, c1, c2, redution=16, kernel_size=7):
        super(CBAM_Attention, self).__init__()
        self.channel_attention = ChannelAttneitn(c1, redution=redution)
        self.spatia_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatia_attention(x)
        return x


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    model = CBAM_Attention(c1=512, c2=512, redution=8, kernel_size=7)
    output=model(input)
    print(model)