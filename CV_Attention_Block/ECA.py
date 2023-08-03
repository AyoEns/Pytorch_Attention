import math

import torch
from torch import nn

class ECA_Attention(nn.Module):
    '''
    与SE注意力机制相似，
    先通过平均池化获得先验权重
    通过自适应卷积核的Conv1D卷积分别获得每个通道的权重
    Sigmoid函数获得权重矩阵
    权重矩阵与输入层相乘获得加强层
    '''
    def __init__(self, c1, b=1, gamma=2):
        super(ECA_Attention, self).__init__()
        kernel_size = int(abs((math.log(c1, 2)) + b))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = (kernel_size - 1) // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.avg_pool(x)
        weight = self.conv(weight.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        weight = self.sigmoid(weight)
        return x * weight.expand_as(x)

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    model = ECA_Attention(c1=512, b=1, gamma=2)
    output=model(input)
    print(model)