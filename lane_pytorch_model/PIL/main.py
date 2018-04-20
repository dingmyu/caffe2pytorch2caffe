import shutil
import time
import argparse
import numpy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torchvision import transforms
from torch.utils.data import DataLoader
from loss import *
from dataloader import *
import lanenet
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', default='')
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--new_length', default=705, type=int)
parser.add_argument('--new_width', default=833, type=int)
parser.add_argument('--label_length', default=177, type=int)
parser.add_argument('--label_width', default=209, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 200)')
parser.add_argument('--resume', default='checkpoints_size', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec = 0


def main():
    global args, best_prec
    args = parser.parse_args()
    print ("Build model ...")
    params = torch.load('/mnt/lustre/dingmingyu/workspace/Pytorch/checkpoints_size/old.pth.tar')
    model = lanenet.Net()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(params['state_dict']) 
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("Saving everything to directory %s." % (args.resume))

    # define loss function (criterion) and optimizer
    criterion = cross_entropy2d
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # data transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)])
    gt_transform = transforms.Compose([ 
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x:x*255),
                        transforms.Lambda(lambda x:numpy.clip(x,0,4))])

    train_data = MyDataset('/mnt/lustre/share/dingmingyu/new_list_lane.txt', args.dir_path, args.new_width, args.new_length,args.label_width,args.label_length,train_transform,gt_transform)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,pin_memory=True)


    for epoch in range(args.start_epoch, args.epochs):
        print 'epoch: ' + str(epoch + 1)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set

        # remember best prec and save checkpoint

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, checkpoint_name, args.resume)



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    #weight_cus = torch.ones(5)
    #weight_cus = weight_cus.cuda()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.float().cuda()
        target = target.long().cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        #print output.size(),target_var.size()
        loss = criterion(output, target_var, size_average=True)

        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))



def save_checkpoint(state, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr / (2 ** (epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # param_group['lr'] = param_group['lr']/2

if __name__ == '__main__':
    main()
