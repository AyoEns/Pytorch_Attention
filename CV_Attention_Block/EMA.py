import torch
from torch import nn

class EMA(nn.Module):
    '''
    EMA跨尺度注意力机制
    '''
    def __init__(self, c1, factor=8):
        super(EMA, self).__init__()
        self.groups = factor    #   分组卷积，默认分8组
        assert c1 // self.groups > 0

        self.softmax = nn.Softmax(-1)
        #  上半分支的验X和Y的池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        #   上半分支的GroupNorm
        self.gn = nn.GroupNorm(c1 // self.groups, c1 // self.groups)
        #   上半分支的1*1卷积
        self.Conv1x1 = nn.Conv2d(c1 // self.groups, c1 // self.groups, 1, 1, padding=0)
        #   下半分支的3x3卷积
        self.Conv3x3 = nn.Conv2d(c1 // self.groups, c1 // self.groups, kernel_size=3, stride=1, padding=1)
        #   分支的池化
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        #   1.调整size。 b,c,h,w -> b*groups, c//g,  h, w
        group_x = x.reshape(b * self.groups, -1, h, w)
        #   2.上半分支
        #   2.1 X和Y方向的池化
        x_h = self.pool_h(group_x)  # b*groups, c//g,  h, 1
                                    #   b*groups, c//g,  -1, w -> permute b*groups, c//g, w, 1
        x_w = self.pool_w(group_x).permute(0,1,3,2) #   注意调整维度
        #   2.2 h和w Cat + 1x1卷积
        hw = self.Conv1x1(torch.cat([x_h, x_w], dim=2)) #  b*groups, c//g,  2*h, 1 -> b*groups, c//g,  2*h, 1 维度没变化
        #   2.3 split
        x_h, x_w = torch.split(hw, [h, w], dim=2) #     b*groups, c//g,  h, 1 and b*groups, c//g,  h, 1
        #   2.4 融合并GN
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0,1,3,2).sigmoid()) # x_w需要调整维度回原来的size 然后两个权重与源输入相乘
        #   x1-> b*groups, c // g, h, w
        #   3 下半分支
        #   3.1 3x3卷积
        x2 = self.Conv3x3(group_x) # -> b, c//g, h, w
        #   3.2 融合阶段
        ###################################################################################################
        #   x11和x12分别代表了中间的融合的Avg和softmax，x11为上半的GN后Avg，x12为下半3x3后的softmax
        #   x1-> b*groups, c // g, h, w -> apg->  b*groups, c // g, 1 -> permute -> b*groupss, 1, c //g
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0,2,1))
        #   x12代表下半分支
        x12 = x2.reshape(b * self.groups, c // self.groups, -1) #   b, c//g, h, w -> b*g, c//g, h*w
        ###################################################################################################
        #   x21和x22分别代表了外面两层的融合的Avg和softmax，x21为下半的3*3后的Avg，x22为上半GN后的softmax
        #   x21
        #   x2 -> b, c//g, h, w -> apg -> b*g, c//g, 1, 1 -> reshape ->  b*g, c//g, 1-> permute -> b*g, 1, c//g
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        #   x22
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # x1-> b*groups, c // g, h, w -> b*g, c//g, h*w
        ###################################################################################################
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b,c,h,w)


if __name__ == '__main__':
    input = torch.randn(1, 512, 640, 640)
    model = EMA(512)
    model(input)
