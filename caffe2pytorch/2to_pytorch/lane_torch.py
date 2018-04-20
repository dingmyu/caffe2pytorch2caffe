
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn
    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input
class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))
class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))
class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


lane_torch = nn.Sequential( # Sequential,
	nn.Conv2d(3,16,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),#AvgPool2d,
	nn.Conv2d(16,16,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(16,8,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(8,4,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(4,4,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(32,32,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(32,32,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(32,32,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(32,64,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(64,128,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,16,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(16,8,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(8,5,(1, 1)),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,16,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(16,8,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,16,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(16,8,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(8,16,(1, 1)),
	nn.Conv2d(8,2,(1, 1)),
)