#   CoordAttention模块
import torch
from torch import nn

class h_sigmoid(nn.Module):
    def __init__(self, inplace):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    #   权重的激活函数换位swish
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=True)

    def forward(self, x):
        return x * self.sigmoid(x)


class CA_Attention(nn.Module):
    def __init__(self, inp, oup, reduciton=32, h_swish_flag=True):
        super(CA_Attention, self).__init__()

        #   特征层进入后先进行在H纬度和W纬度上池化得到特征层
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        #   中间层对pool_h和pool_w进行Concat后，进行1X1的卷积
        mip = max(8, int(inp // reduciton))
        self.Conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.Bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish() if h_swish_flag else nn.ReLU(inplace=True)

        #   通道分离后又分别进行2次1X1的卷积
        self.Conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.Conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #   前向推理
        #   identity待加强的特征层
        identity = x
        #   首先进行在h，w纬度上池化得到初步权重矩阵

        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)    #   注意这里w纬度池化后的纬度变化

        #   二阶段：Concat + 1X1Conv
        y = torch.cat((x_h, x_w), dim=2)    #   在dim=2即在Batch纬度上堆叠
        y = self.act(self.Bn1(self.Conv1(y)))

        #   三阶段：通道分离 + 分组1X1卷积
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)       #   注意这里w纬度池化后的纬度变化
        weight_h = self.Conv_h(x_h).sigmoid()
        weight_w = self.Conv_w(x_w).sigmoid()

        #   权重矩阵加权
        out = identity * weight_h * weight_w
        return out


if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    CA_model = CA_Attention(inp=512, oup=512, h_swish_flag=True)
    output = CA_model(input)
    print(CA_model)



