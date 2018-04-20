import shutil
import time
import argparse
import numpy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from loss import *
from dataloader import *
import lanenet
import os
import torchvision as tv
from torch.autograd import Variable
from PIL import Image
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--img_list', dest='img_list', default='/mnt/lustre/dingmingyu/data/test_pic/list.txt',                         help='the test image list', type=str)
parser.add_argument('--img_dir', dest='img_dir', default='/mnt/lustre/dingmingyu/data/test_pic/',
                        help='the test image dir', type=str)
parser.add_argument('--model_path', dest='model_path', default='/mnt/lustre/dingmingyu/workspace/Pytorch/checkpoints_size/011_checkpoint.pth.tar',
                        help='the test model', type=str)

def main():
    global args
    args = parser.parse_args()
    print ("Build model ...")
    model = lanenet.Net()
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load(args.model_path)['state_dict']
    model.load_state_dict(state)
    model.eval()    

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
    f = open(args.img_list)
    ni = 0
    for line in f:
        line = line.strip()
        arrl = line.split(" ") 
        image = tv.datasets.folder.default_loader(args.img_dir + arrl[0])
        image = image.resize((833,705),Image.ANTIALIAS)
        #image = cv2.imread(args.img_dir + arrl[0], -1)
        #image = cv2.resize(image, (732,704), interpolation = cv2.INTER_NEAREST)
        image = train_transform(image)
        print image.shape
        image = image.unsqueeze(0)
        image = Variable(image.cuda(0), volatile=True)
    	output = model(image)
    	#output = F.log_softmax(output, dim=1)
    	prob = output.data[0].max(0)[1].cpu().numpy()
        print output.size()
    	#print output.max(),type(output)

        probAll = np.zeros((prob.shape[0], prob.shape[1], 3), dtype=np.float)
        prob1 = prob.copy()
        prob1[prob1 != 1] = 0
        prob2 = prob.copy()
        prob2[prob2 != 2] = 0
        prob3 = prob.copy()
        prob3[prob3 != 3] = 0
        prob4 = prob.copy()
        prob4[prob4 != 4] = 0
        probAll[:,:,0] += prob1 # line 1
        probAll[:,:,1] += prob2 # line 2
        probAll[:,:,2] += prob3 # line 3

        probAll[:,:,0] += prob4 # line 4
        probAll[:,:,1] += prob4 # line 4
        probAll[:,:,2] += prob4 # line 4

        probAll = np.clip(probAll * 255, 0, 255)

        test_img = cv2.imread(args.img_dir + arrl[0], -1)
       # probAll = cv2.resize(probAll, (224,192), interpolation = cv2.INTER_NEAREST)
        test_img = cv2.resize(test_img, (209,177))

        ni = ni + 1
        test_img = np.clip(test_img + probAll, 0, 255).astype('uint8')
        cv2.imwrite(args.img_dir + 'prob/test_' + str(ni) + '_lane.png', test_img)
        print('write img: ' + str(ni+1))
    f.close()

if __name__ == '__main__':
    main()

