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
import lanenet
import os
import torchvision as tv
from torch.autograd import Variable
from PIL import Image
import cv2
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth

parser = argparse.ArgumentParser()
parser.add_argument('--img_list', dest='img_list', default='/mnt/lustre/sunpeng/ADAS_lane_prob_IoU_evaluation/GMXL/new_camera1212_test.txt',
                            help='the test image list', type=str)

parser.add_argument('--img_dir', dest='img_dir', default='/mnt/lustre/drive_data/adas-video/record/',
                        help='the test image dir', type=str)
parser.add_argument('--gtpng_dir', dest='gtpng_dir', default='/mnt/lustre/drive_data/adas-video/record/',
                        help='the test png label dir', type=str)

parser.add_argument('--model_path', dest='model_path', default='checkpoints/017_checkpoint.pth.tar',
                        help='the test model', type=str)

parser.add_argument('--prethreshold', dest='prethreshold', default=0.4,
                        help='preset bad threshold', type=float)

def iou_one_frame(rprob, gtpng, bias_threshold):
    iou = [0, 0, 0, 0]   
    gt1 = gtpng.copy()
    gt1[gt1 > 1] = 0
    gt1[gt1 < 1] = 0
    gt2 = gtpng.copy()
    gt2[gt2 > 2] = 0
    gt2[gt2 < 2] = 0
    gt3 = gtpng.copy()
    gt3[gt3 > 3] = 0
    gt3[gt3 < 3] = 0
    gt4 = gtpng.copy()
    gt4[gt4 > 4] = 0
    gt4[gt4 < 4] = 0
    for i in range(1,5):
        prob = rprob.copy()
        prob[prob > i] = 0
        prob[prob < i] = 0
        max_iou = -100
        if len(prob[prob == i]) > 0:        
            for gt in [gt1, gt2, gt3, gt4]:      
                cross = np.multiply(gt, prob)
                cross[cross >= 1] = 1
                unite = gt + prob
                unite[unite >= 1] = 1
                union = np.sum(unite)
                inter = np.sum(cross)
                if (union != 0):
                    small_iou = inter * 1.0 / union
                    if (np.sum(gt) == 0):
                        small_iou = -1
                else:
                    small_iou = -10
                if small_iou > max_iou:
                    max_iou = small_iou
        iou[i-1] = max_iou
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

    mean = [104, 117, 123]
    f = open(args.img_list)
    ni = 0
    count_gt = [0,0,0,0]
    total_iou = [0,0,0,0]
    total_iou_big = [0,0,0,0]
    for line in f:
        #if ni > 3:
        #    break
        line = line.strip()
        arrl = line.split(" ")
    
        #gtlb = cv2.imread(args.gtpng_dir + arrl[1], -1)
        gtlb = cv2.imread(args.gtpng_dir + arrl[1], -1)
        #print gtlb.shape
        image = cv2.imread(args.img_dir + arrl[0])
        image = cv2.resize(image,(833,705)).astype(np.float32)
        image -= mean
        image = image.transpose(2, 0, 1)
        #image = cv2.imread(args.img_dir + arrl[0], -1)
        #image = cv2.resize(image, (732,704), interpolation = cv2.INTER_NEAREST)
        #print image.shape
        image = torch.from_numpy(image).unsqueeze(0)
        image = Variable(image.float().cuda(0), volatile=True)
        t = time.time()
        output, embedding = model(image)
        print time.time() - t
        output = torch.nn.functional.softmax(output[0],dim=0)
        prob = output.data.cpu().numpy()
        embedding = embedding.data.cpu().numpy()

        prob[prob >= args.prethreshold] = 1.0
        prob[prob < args.prethreshold] = 0
        embedding = embedding[0,:,:,:].transpose((1, 2, 0))
        #print prob.shape
        mylist = []
        indexlist = []
        for i in range(embedding.shape[0]):
            for j in range(embedding.shape[1]):
                if prob[1][i][j] > 0:
                    mylist.append(embedding[i,j,:])
                    indexlist.append((i,j))
        if not mylist:
            continue
        mylist = np.array(mylist)
       # bandwidth = estimate_bandwidth(mylist, quantile=0.3, n_samples=100, n_jobs = 8)
       # print bandwidth
        estimator = MeanShift(bandwidth=1, bin_seeding=True)
#        estimator = KMeans(n_clusters = 4)
        #estimator = AffinityPropagation(preference=-0.4, damping = 0.5)
        t = time.time()
        estimator.fit(mylist)
        print time.time() - t
        for i in range(4):
            print len(estimator.labels_[estimator.labels_==i])
        print len(np.unique(estimator.labels_)),'~~~~~~~~~~~~~~~~'
        new_prob = np.zeros((embedding.shape[0],embedding.shape[1]),dtype=int)
        for index, item in enumerate(estimator.labels_):
            if item < 4:
                new_prob[indexlist[index][0]][indexlist[index][1]] = item + 1

        gtlb = cv2.resize(gtlb, (prob.shape[2], prob.shape[1]),interpolation = cv2.INTER_NEAREST)
        iou = iou_one_frame(gtlb, new_prob, args.prethreshold)

#        print gtlb, new_prob, gtlb.max(), new_prob.max()

        print('IoU of ' + str(ni) + ' '+ arrl[0] + ': ' + str(iou))
        for i in range(0,4):
            if iou[i] >= 0:
                count_gt[i] = count_gt[i] + 1
                total_iou[i] = total_iou[i] + iou[i]
        ni += 1
    mean_iou = np.divide(total_iou, count_gt)
    print('Image numer: ' + str(ni))
    print('Mean IoU of four lanes: ' + str(mean_iou))
    print('Overall evaluation: ' + str(mean_iou[0] * 0.2 + mean_iou[1] * 0.3 + mean_iou[2] * 0.3 + mean_iou[3] * 0.2))

    f.close()

if __name__ == '__main__':
    main()

