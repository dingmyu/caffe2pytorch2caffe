## IOU:
* weight 2:1
    * binary_seg:	0.6308
    * ins_seg(meanshift):	0.6358
    * reg\_ins\_seg(meanshift):	0.6253
    * cls\_ins\_seg(k-means):	0.6197

* weight 1:1
    * binary_seg:	0.6326
    * ins_seg(meanshift):	0.6245
    * reg\_ins\_seg(meanshift):	0.6352
    * cls\_ins\_seg(k-means):	0.6116

## ins_accuracy
* ins(meanshift):	0.8316(include 0), 0.8158(exclude 0)
* cls:	0.9130

## time
* seg_net:	5ms
* ins\_seg\_net:	6.5ms
* meanshift:	18ms(python)
* kmeans:	11.5ms(python)
