#--coding:utf-8--
import torch.nn as nn
import torch
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, )

#import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()
        
        self.m0 = nn.Conv2d(3,16,(3, 3),(2, 2),(1, 1))
        self.m3 = nn.Conv2d(16,16,(3, 3),(1, 1),(1, 1))
        self.m5 = nn.Conv2d(16,8,(3, 3),(1, 1),(1, 1))
        self.m7 = nn.Conv2d(8,4,(3, 3),(1, 1),(1, 1))
        self.m9 = nn.Conv2d(4,4,(3, 3),(1, 1),(1, 1))
        self.m11 = nn.Conv2d(32,32,(3, 3),(2, 2),(1, 1))
        self.m13 = nn.Conv2d(32,32,(3, 3),(2, 2),(1, 1), )
        self.m15 = nn.Conv2d(32,32,(3, 3),(1, 1),(1, 1), )
        self.m17 = nn.Conv2d(32,64,(3, 3),(2, 2),(1, 1), )
        self.m19 = nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1), )
        self.m21 = nn.Conv2d(64,128,(3, 3),(2, 2),(1, 1), )
        self.m23 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1), )
        self.m25 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1), )
        self.m27 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1), )
        self.m29 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1), )
        self.m31 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1), )
        self.m33 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1), )
        self.m35 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1), )
        self.m37 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1), )
        self.m39 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1), )
        self.m41 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1), )
        self.m43 = nn.Conv2d(128,16,(3, 3),(1, 1),(1, 1), )
        self.m45 = nn.Conv2d(16,8,(3, 3),(1, 1),(1, 1), )
        self.m47 = nn.Conv2d(8,5,(1, 1))


    def forward(self, x): 
        x = nn.ReLU()(self.m0(x))
        x = nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True)(x)
        x1 = nn.ReLU()(self.m3(x))
        x2 = nn.ReLU()(self.m5(x1))
        x3 = nn.ReLU()(self.m7(x2))
        x4 = nn.ReLU()(self.m9(x3))
        x = torch.cat([x1, x2, x3, x4], dim = 1)
        x = nn.ReLU()(self.m11(x))
        x = nn.ReLU()(self.m13(x))
        x = nn.ReLU()(self.m15(x))
        x = nn.ReLU()(self.m17(x))
        x = nn.ReLU()(self.m19(x))
        x = nn.ReLU()(self.m21(x))
        x = nn.ReLU()(self.m23(x))
        x = nn.ReLU()(self.m25(x))
        x = nn.ReLU()(self.m27(x))
        x = nn.ReLU()(self.m29(x))
        x = nn.ReLU()(self.m31(x))
        x = nn.ReLU()(self.m33(x))
        x = nn.ReLU()(self.m35(x))
        x = nn.ReLU()(self.m37(x))
        x = nn.ReLU()(self.m39(x))
        x = nn.ReLU()(self.m41(x))
	x = nn.Upsample(size=(45,53),mode='bilinear')(x)
        x = nn.ReLU()(self.m43(x))
	x = nn.Upsample(size=(177,209),mode='bilinear')(x)
        x = nn.ReLU()(self.m45(x))
        x = self.m47(x)
        return x

#net = Net()
#print(net)
