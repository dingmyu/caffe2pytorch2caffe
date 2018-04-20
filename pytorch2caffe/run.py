import torch
from torch.autograd import Variable
import torchvision

import sys

sys.path.append('/mnt/lustre/dingmingyu/software/core/python')
import caffe

caffe.set_mode_gpu()
caffe.mpi_init()

import os
from pytorch2caffe import pytorch2caffe, plot_graph
import lanenet
state_dict =torch.load('lane_torch.pth')
model = lanenet.Net()
model_dict = model.state_dict()
pretrained_dict = {'m'+k: v for k, v in state_dict.items() if 'm'+k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)


model.eval()
print(model)

input_var = Variable(torch.rand(1, 3, 705, 836))
output_var = model(input_var)

output_dir = 'demo'
# plot graph to png
#plot_graph(output_var, os.path.join(output_dir, 'lanenet.dot'))

pytorch2caffe(input_var, output_var, 
              os.path.join(output_dir, 'pytorch2caffe.prototxt'),
              os.path.join(output_dir, 'pytorch2caffe.caffemodel'))
