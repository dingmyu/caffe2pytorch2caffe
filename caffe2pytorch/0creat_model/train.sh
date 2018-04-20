#!/bin/sh

#export PATH=/mnt/lustre/share/cuda-7.5/bin:$PATH
#export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-7.5/lib:/mnt/lustre/#share/cuda-7.5/lib64:$LD_LIBRARY_PATH

#CAFFE_DIR=/mnt/lustre/roadExtraction_share/all_backup/chongruo_11/caffe_multigpu
#CAFFE_DIR=/mnt/lustre/roadExtraction_share/caffe_multigpu
#CAFFE_BIN=${CAFFE_DIR}/build/tools/caffe.bin
#export LD_LIBRARY_PATH=/mnt/lustre/share/Mog/lib/lib:$LD_LIBRARY_PATH
CAFFE_DIR=/mnt/lustre/dingmingyu/software/core
#CAFFE_DIR=/mnt/lustre/roadExtraction_share/caffe_multigpu
CAFFE_BIN=${CAFFE_DIR}/build/tools/caffe


################################################## slurm
#now=$(date +"%Y%m%d_%H%M%S")
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun --mpi=pmi2 --gres=gpu:4 -n4 --ntasks-per-node=2 /mnt/lustre/yanshengen/sensenet/example/build/tools/caffe train --solver=resnet200_solver.prototxt

#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 GLOG_logtostderr=1 srun --mpi=pmi2 --gres=gpu:4 -n6 --ntasks-per-node=2 ${CAFFE_BIN} train --solver=solver0811.prototxt 
#--kill-on-bad-exit=1
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 GLOG_logtostderr=1 \
srun --mpi=pmi2 --gres=gpu:4 -n1 --ntasks-per-node=4  --partition=bj11test --kill-on-bad-exit=1 --job-name=3on1 \
${CAFFE_BIN} train --solver=solver.prototxt \
--weights=model.caffemodel \
2>&1 | tee train.log &
#--snapshot=model/lane_iter_450000.solverstate \



#--snapshot=snapshot/resnet_half_iter_49000.solverstate
#--snapshot=model/gf2mapa_1st_abc_part_iter_60000.solverstate
#GLOG_logtostderr=1 CUDA_VISIBLE_DEVICES=0,2,5,6,7


