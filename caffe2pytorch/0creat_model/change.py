import sys
sys.path.insert(0, '/mnt/lustre/dingmingyu/software/core/python')
import caffe
import numpy as np

if __name__ == '__main__':

    gpu_id = 0
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    ## you need to prepare 
    ## 1, combined_net.prototxt; 2, net_a.caffemodel, 3, net_b.caffemodel, 4, net_b.prototxt
    ## first load net_a.caffemodel into combined model
    ## then load net_b.caffemodel into combined model by matching the layer name
    prototxt = 'deploy_crt.prototxt' #combined net proto
    caffemodel = 'random.caffemodel' #lka net
    caffe.mpi_init()
    #net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net = caffe.Net(prototxt, caffe.TEST)
    prototxt2 = 'deploy_crt.prototxt' #fcw net proto
    caffemodel2 = 'best.caffemodel' #fcw net
    net2 = caffe.Net(prototxt2, caffemodel2, caffe.TEST)
    filename = 'model_new.caffemodel'
    # apply bbox regression normalization on the net weights
    flag = False

    for layer_name, params in net2.params.iteritems():
	if layer_name:
	    print layer_name
  	    flag &= True
	    net.params[layer_name][0].data[...] = net2.params[layer_name][0].data
	    if len(params) >= 2:
		print layer_name + ' has 2 bias'
	        net.params[layer_name][1].data[...] = net2.params[layer_name][1].data
	    if len(params) >= 3:
		print layer_name + ' has 3 blobs'
		net.params[layer_name][2].data[...] = net2.params[layer_name][2].data
  	    if len(params) >= 4:
		print layer_name + ' has 4 blobs'
                net.params[layer_name][3].data[...] = net2.params[layer_name][3].data
            if len(params) >= 5:
                print layer_name + ' has 5 blobs'
                net.params[layer_name][4].data[...] = net2.params[layer_name][4].data
            if len(params) >= 6:
                print layer_name + ' has 6 blobs'
                net.params[layer_name][5].data[...] = net2.params[layer_name][5].data
            if len(params) >= 7:
                print layer_name + ' has more than 6 blobs, ' + str(len(params))
                flag = False

    if flag:
        print 'something wrong, no weights copied'
    else:
        print 'saving...'
        net.save(str(filename))
        print 'complete'

