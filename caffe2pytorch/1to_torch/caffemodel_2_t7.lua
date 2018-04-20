require 'loadcaffe'
require 'xlua'
require 'optim'

-- modify the path 

prototxt = 'deploy_crt.prototxt'
binary = 'model_new.caffemodel'

net = loadcaffe.load(prototxt, binary, 'cudnn')
net = net:float() -- essential reference https://github.com/clcarwin/convert_torch_to_pytorch/issues/8
print(net)

torch.save('lane_torch.t7', net)
