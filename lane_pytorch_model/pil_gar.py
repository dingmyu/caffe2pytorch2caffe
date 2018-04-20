%matplotlib inline
pic_name = '/mnt/lustre/drive_data/adas-video/record/GMXL1207/GMXL_Img/20171103/Random/Sunny/Night/20170927204720.MP4/00000700.jpg'
gt_name = '/mnt/lustre/drive_data/adas-video/record/GMXL1207/Gt_png/GMXL_Img/20171103/Random/Sunny/Night/20170927204720.MP4/00000700.png'
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image 
pil_gt = Image.open(gt_name)
pil_gt = pil_gt.resize((836,705), Image.NEAREST)
cv_gt = cv2.imread(gt_name,0)
cv_gt = cv2.resize(cv_gt,(836,705),interpolation = cv2.INTER_NEAREST)

a =  cv_gt.astype(np.float32) - np.array(pil_gt).astype(np.float32)
print np.abs(a).sum()

cv_pic = cv2.imread(pic_name)
cv_pic = cv2.resize(cv_pic,(836,705),interpolation = cv2.INTER_LINEAR)
cv_gt = np.clip(cv_gt.astype(np.float32)*255,0,100)
cv_gt = np.expand_dims(cv_gt, axis=2).repeat(3,2)
new_image_cv = np.clip(cv_gt + cv_pic,0,250)
image_cv = Image.fromarray(new_image_cv[:,:,::-1].astype('uint8'))


pil_gt = np.clip(np.array(pil_gt).astype(np.float32)*255,0,100)
pil_pic = Image.open(pic_name).convert('RGB')
pil_pic = np.array(pil_pic.resize((836,705),Image.ANTIALIAS)).astype(np.float32)
pil_gt = np.expand_dims(pil_gt, axis=0).repeat(3,0)
pil_gt= pil_gt.transpose(1,2,0)
new_image = np.clip(pil_pic + pil_gt,0,250)
image = Image.fromarray(new_image.astype('uint8'))
image_cv


#plt.imshow(new_image_cv[:,:,::-1].astype('uint8'))
#plt.show()

#cv2.imwrite('cv2.jpg',new_image_cv)
#cv2.imshow('result',new_image_cv)
#cv2.waitKey(0)
