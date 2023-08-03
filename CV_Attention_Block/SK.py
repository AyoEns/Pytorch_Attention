import torch
from torch import nn
from collections import OrderedDict
# https://blog.csdn.net/CharmsLUO/article/details/109784124?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169107338416800226558164%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=169107338416800226558164&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-109784124-null-null.142^v92^koosearch_v1&utm_term=SK%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&spm=1018.2226.3001.4187

class SK_Attention(nn.Module):
    #   SK注意力机制
    def __init__(self, c1, kernel=[1, 3, 5, 7], redution=16, guoups=1, L=32):
        super(SK_Attention, self).__init__()
        self.d = max(L, c1 // redution)
        self.Split_convs = nn.ModuleList([])
        #   第一阶段，特征层输入，经过卷积核3 5 7的卷积生成不同感受野的特征层
        for k in kernel:
            self.Split_convs.append(
                nn.Sequential(
                    nn.Conv2d(c1, c1, kernel_size=k, padding=k//2, groups=guoups),
                    nn.BatchNorm2d(c1),
                    nn.ReLU(inplace=True)
                )
            )

        #   二阶段，特征层沿着W和H展开，通过全连接层压缩为c1 * d
        self.fc = nn.Linear(c1, self.d)

        #   三阶段，不同特征层经过各自的全连接层，生产权重矩阵
        self.fcs = nn.ModuleList([])
        for i in range(len(kernel)):
            self.fcs.append(nn.Linear(self.d, c1))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        b, c,_,_ = x.size()
        conv_outs = []

        #   第一阶段：分离
        for conv in self.Split_convs:
            conv_outs.append(conv(x))

        #   沿着垂直方向堆叠
        feats = torch.stack(conv_outs, 0) # k, bs, channel, h, w

        #   二阶段：融合，Add 三个特征层相加
        U = sum(conv_outs)  #   bs, c, h, w

        #   展平mean(-1)沿着最后一维，两次即沿着W和H展平
        S = U.mean(-1).mean(-1) # bs, c
        #   全连接层压缩成 通道数为c1 * 1 * 1的特征层
        Z = self.fc(S)

        #   三阶段：不同通道的权重
        #   都对Z进行全连接
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            # View相当于numpy中reshape，沿着 b，c纬度进行纬度划分，划分后尺寸 (b, c, 1, 1)
            weights.append(weight.view(b, c, 1, 1)) # bs, channel
        attention_weights = torch.stack(weights, 0) #  fc, b, c, 1, 1
        attention_weights = self.softmax(attention_weights) #   分别各自进行softmax求出权重矩阵

        #   权重融合
        #    (attention_weights * feats)与各自原先位置的特征层U1 U2 U3相乘加强，后Add生成最后特征层
        V = (attention_weights * feats).sum(0)
        return V

if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    model = SK_Attention(c1=512, kernel=[3, 5, 7], redution=8)
    output = model(input)
    print(model)
