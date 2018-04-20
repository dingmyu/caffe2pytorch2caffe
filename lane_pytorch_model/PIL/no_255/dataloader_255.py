import torch.utils.data as data
import os
from PIL import Image
import numpy as np
#/mnt/lustre/share/dingmingyu/new_list_lane.txt

class MyDataset(data.Dataset):
    def __init__(self, file, dir_path, new_width, new_height, label_width, label_height):
        imgs = []
        fw = open(file, 'r')
        lines = fw.readlines()
        for line in lines:
            words = line.strip().split()
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.dir_path = dir_path
        self.height = new_height
        self.width = new_width
        self.label_height = label_height
        self.label_width = label_width

    def __getitem__(self, index):
        path, label = self.imgs[index]
        path = os.path.join(self.dir_path, path)
        img = Image.open(path).convert('RGB')
        img = img.resize((self.width, self.height), Image.ANTIALIAS)
        img = np.array(img).astype('int64')
        img -= [104, 117, 123]
        img = img.transpose(2, 0, 1)
        gt = Image.open(label).split()[0]
        gt = gt.resize((self.label_width, self.label_height), Image.NEAREST)  
        gt = np.array(gt, dtype=np.uint8)
        #gt = np.clip(gt,0,4)
        return img, gt

    def __len__(self):
        return len(self.imgs)