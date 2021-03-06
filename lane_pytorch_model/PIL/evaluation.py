import shutil
import time
import argparse
import numpy as np
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
parser.add_argument('--img_list', dest='img_list', default='/mnt/lustre/sunpeng/ADAS_lane_prob_IoU_evaluation/GMXL/new_camera1212_test.txt',
                            help='the test image list', type=str)

parser.add_argument('--img_dir', dest='img_dir', default='/mnt/lustre/drive_data/adas-video/record/',
                        help='the test image dir', type=str)
parser.add_argument('--gtpng_dir', dest='gtpng_dir', default='/mnt/lustre/drive_data/adas-video/record/',
                        help='the test png label dir', type=str)

parser.add_argument('--model_path', dest='model_path', default='/mnt/lustre/dingmingyu/workspace/Pytorch/checkpoints2/007_checkpoint.pth.tar',
                        help='the test model', type=str)


def iou_one_frame(gtpng, prob):
    iou = [0, 0, 0, 0]   
    for i in range(1,5):
        gt = gtpng.copy()
        gt[gt > i] = 0
        gt[gt < i] = 0
        rprob = prob.copy()
        rprob[rprob > i] = 0
        rprob[rprob < i] = 0
        cross = np.multiply(gt, rprob)
        cross[cross >= 1] = 1
        unite = gt + rprob
        unite[unite >= 1] = 1
        union = np.sum(unite)
        inter = np.sum(cross)
        if (union != 0):
            iou[i-1] = inter * 1.0 / union
            if (np.sum(gt) == 0):
                iou[i-1] = -1
        else:
            iou[i-1] = -10
    return iou


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
    count_gt = [0, 0, 0, 0]
    total_iou = [0, 0, 0, 0]
    total_iou_big = [0, 0, 0, 0]
    for line in f:
        line = line.strip()
        arrl = line.split(" ")
    
        #gtlb = cv2.imread(args.gtpng_dir + arrl[1], -1)
        gtlb = Image.open(args.gtpng_dir + arrl[1])
        #print gtlb.shape
        image = tv.datasets.folder.default_loader(args.img_dir + arrl[0])
        image = image.resize((833,705),Image.ANTIALIAS)
        #image = cv2.imread(args.img_dir + arrl[0], -1)
        #image = cv2.resize(image, (732,704), interpolation = cv2.INTER_NEAREST)
        image = train_transform(image)
        image = image.unsqueeze(0)
        image = Variable(image.cuda(0), volatile=True)
        output = model(image)
        print output.size()
        #output = F.log_softmax(output, dim=1)
        prob = output.data[0].max(0)[1].cpu().numpy()#.transpose(1,0)
        #prob.transpose((1,0))
        gtlb = np.array(gtlb.resize((prob.shape[1], prob.shape[0]),Image.ANTIALIAS))
        #print gtlb.shape
        #print gtlb[100:110,100:110]
        #print prob[100:110,100:110] 
        #print output.max(),type(output)
        iou = iou_one_frame(gtlb, prob)
        print('IoU of ' + str(ni) + ' '+ arrl[0] + ': ' + str(iou))
        for i in range(0,4):
            if iou[i] >= 0:
                count_gt[i] = count_gt[i] + 1
                total_iou[i] = total_iou[i] + iou[i]

    mean_iou = np.divide(total_iou, count_gt)
    print('Image numer: ' + str(ni))
    print('Mean IoU of four lanes: ' + str(mean_iou))
    print('Overall evaluation: ' + str(mean_iou[0] * 0.2 + mean_iou[1] * 0.3 + mean_iou[2] * 0.3 + mean_iou[3] * 0.2))

    f.close()

if __name__ == '__main__':
    main()

