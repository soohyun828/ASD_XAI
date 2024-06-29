#!/bin/bash

#SBATCH --job-name asd_annotation_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=25G
#SBATCH --time 1-00:00:0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y5
#SBATCH -o /data/psh68380/repos/ASD_capstone/sbatch_log/asd_annotation_%A-%x.out
#SBATCH -e /data/psh68380/repos/ASD_capstone/sbatch_log/asd_annotation_%A-%x.err
echo $PWD
echo $SLURMD_NODENAME
current_time=$(date "+%Y%m%d-%H:%M:%S")

echo $current_time
export MASTER_PORT=16445

# batch_size can be adjusted according to number of GPUs
# this script is for 2 GPUs (1 nodes x 2 GPUs)
# --data_root "/local_datasets/ai_hub_sketch_mw/01/train"
python -u /data/psh68380/repos/ASD_capstone/main.py \
--data_root "/local_datasets/ASD/asd_ver2_all_5folds_annotation" \
--num_epochs 30 \
--batch_size 4 \
--learning_rate 0.0001 \
--image_model "efficientnetb0" \
--part_model "linear"

    
echo "Job finish"
exit 0