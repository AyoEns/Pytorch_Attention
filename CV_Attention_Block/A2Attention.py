import math
from torch.nn import init
import torch
from torch import nn
from torch.nn import functional as F

class DoubleAttention(nn.Module):

    def __init__(self, c1, c2=128, c3=128, reconstruct=True):
        super(DoubleAttention, self).__init__()
        self.in_channels = c1
        self.reconstruct = reconstruct
        self.c_m = c2
        self.c_n = c3
        self.ConvA = nn.Conv2d(self.in_channels, self.c_m, 1)
        self.ConvB = nn.Conv2d(self.in_channels, self.c_n, 1)
        self.ConnV = nn.Conv2d(self.in_channels, self.c_n, 1)
        if self.reconstruct:
            self.Conv_reconstruct = nn.Conv2d(self.c_m, self.in_channels, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.0001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, x):
        b, c, h, w = x.shape
        assert  c == self.in_channels
        # 开头三个卷积
        # A代表分支
        # B代表特征
        # V代表分布
        A = self.ConvA(x)
        B = self.ConvB(x)
        V = self.ConnV(x)


        tempA = A.view(b, self.c_m, -1) # b,c_m,h*w
        attention_map = F.softmax(B.view(b, self.c_n, -1))# b,c_n,h*w
        attention_vectors = F.softmax(V.view(b, self.c_n, -1))# b,c_n,h*w
        # Step1 特征提取 #  保持第一纬度不变相乘
        global_descriptors = torch.bmm(tempA, attention_map.permute(0,2,1)) # [b,c_m,h*w] * [b, h*w, c_n]
        # Step2 特征分布 #   同意矩阵相乘
        tmpZ = global_descriptors.matmul(attention_vectors)
        tmpZ = tmpZ.view(b, self.c_m, h, w)
        if self.reconstruct:
            tmpZ = self.Conv_reconstruct(tmpZ)

        return tmpZ


if __name__ == '__main__':
    input=torch.randn(100,512,6,6)
    a2 = DoubleAttention(512)
    output=a2(input)
    print(output.shape)








