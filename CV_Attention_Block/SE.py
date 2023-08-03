import torch
from torch import nn
from torch.nn import init

class SE_Attention(nn.Module):
    #   轻量级的注意力机制 SE-Attention
    '''
    对输入进来的特征层进行全局平均池化。
    然后进行两次全连接，第一次全连接神经元个数较少，第二次全连接神经元个数和输入特征层相同。
    在完成两次全连接后，我们再取一次Sigmoid将值固定到0-1之间，此时我们获得了输入特征层每一个通道的权值（0-1之间）。
    '''
    def __init__(self, c1, c2, reduction=16):
        super(SE_Attention, self).__init__()
        #   池化获得权重
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #   连续两层FC层 可以用2次1X1卷积代替
        self.Conv1 = nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.Conv2 = nn.Conv2d(c1 // reduction, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.act_func = nn.Sigmoid()

    def forward(self, x):
        identity = x
        weight = self.avgpool(x)
        weight = self.relu(self.Conv1(weight))
        weight = self.act_func(self.Conv2(weight))
        out = identity * weight.expand_as(x)
        return out

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    se = SE_Attention(c1=512, c2=512, reduction=8)
    output=se(input)
    print(se)