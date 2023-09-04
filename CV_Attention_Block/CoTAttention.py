import torch
from torch import nn
from torch.nn import functional as F

class CoTAttention(nn.Module):
    '''
    CoT Attention
    '''
    def __init__(self, c1, kernel_size=3):
        super(CoTAttention, self).__init__()
        self.dim = c1
        self.kernel_size = kernel_size

        #   QKV
        #   K
        self.key_embed = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size, padding=kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )

        # V
        self.value_embed = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1)
        )

        factor = 4

        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * c1, 2*c1//factor, 1, bias=False),
            nn.BatchNorm2d(2 * c1 // factor),
            nn.ReLU(),
            nn.Conv2d(2 * c1 // factor, kernel_size * kernel_size * c1, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        #   第一阶段 求解K和V值
        k1 = self.key_embed(x) # b,c,h,w -> b,c,h,w
        v = self.value_embed(x).view(bs, c, -1) # b,c,h,w -> b,c,h*w

        #   第二阶段融合， Q值（原输入）与k1堆叠
        y = torch.cat([k1, x], dim=1) # b,c,h,w -> b,2*c,h,w
        #   经过attentio_embed层
        att_weights = self.attention_embed(y) # b,2*c,h,w -> b, c*k*k,h,w
        att_weights = att_weights.reshape(bs, c, self.kernel_size*self.kernel_size, h, w) #  b,c*k*k,h,w -> b,c,k*k,h,w
        # b,c,k*k,h,w (mean求平均压缩了第三维度)-> b,c,h,w -> b,c,h*w
        att_weights = att_weights.mean(dim=2, keepdim=False).view(bs, c, -1)
        #   第三阶段求解权重
        k2 = F.softmax(att_weights, dim=-1) * v # (att=[b,c,h*w]) * (v=[b,c,h*w])
        k2 = k2.view(bs, c, h, w)   #   信息交互后再reshape回原样b,c,h,w

        return k1 + k2

if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    CoT = CoTAttention(c1=512)
    output = CoT(input)
    print(output.shape)