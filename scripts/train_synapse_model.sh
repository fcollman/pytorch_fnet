#!/bin/bash -x

DATASET=${1:-dna}
N_ITER=500
BUFFER_SIZE=1
BATCH_SIZE=24
RUN_DIR="saved_models/${DATASET}"
PATH_DATASET_ALL_CSV="data/csvs/${DATASET}.csv"
PATH_DATASET_TRAIN_CSV="data/csvs/${DATASET}/train.csv"
GPU_IDS=${2:-0}

cd $(cd "$(dirname ${BASH_SOURCE})" && pwd)/..

python scripts/python/split_dataset.py ${PATH_DATASET_ALL_CSV} "data/csvs" --train_size 0.75 -v
python train_model.py \
       --nn_module fnet_nn_3d \
       --n_iter ${N_ITER} \
       --path_dataset_csv ${PATH_DATASET_TRAIN_CSV} \
       --buffer_size ${BUFFER_SIZE} \
       --buffer_switch_frequency 2000000 \
       --batch_size ${BATCH_SIZE} \
       --path_run_dir ${RUN_DIR} \
       --gpu_ids ${GPU_IDS} \
       --n_in_channels 4\
       --patch_size 32 32 32 \
       --class_dataset MultiTiffDataset \
       --transform_signal 'fnet.transforms.ReflectionPadder3d((7,0,1))' 'fnet.transforms.Cropper((0,1,0))' \
       --transform_target 'fnet.transforms.ReflectionPadder3d((7,0,1))' 'fnet.transforms.Cropper((0,1,0))'
