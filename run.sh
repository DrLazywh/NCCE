#!/bin/bash
root_dir=/home/wh/Text_based_Reid/mydatasets
select_ratio=0.3
batch_size=256
weight=3.0

for DATASET_NAME in "ICFG-PEDES" "RSTPReid" "CUHK-PEDES"; do
  for noisy_rate in 0.0 0.2 0.5; do
    CUDA_VISIBLE_DEVICES=0 \
    python train.py \
      --noisy_rate $noisy_rate \
      --noisy_file ./noiseindex/${DATASET_NAME}_${noisy_rate}.npy \
      --name RDE \
      --sampler identity \
      --img_aug \
      --txt_aug \
      --cross_id 1 \
      --batch_size $batch_size \
      --select_ratio $select_ratio \
      --root_dir $root_dir \
      --output_dir new_best_logs3_retpreid \
      --dataset_name $DATASET_NAME \
      --loss_names TAL+sr${select_ratio}_n${noisy_rate}_cross1_batch${batch_size}_weight${weight}_${DATASET_NAME} \
      --num_epoch 60 \
      --weight_id $weight
  done
done
