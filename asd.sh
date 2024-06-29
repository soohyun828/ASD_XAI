#!/bin/bash
#SBATCH -p batch_ce_ugrad 
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=25G
#SBATCH --time=1-00:00:0

socket_ifname=$(cat /etc/hosts | grep $(hostname) | grep -Eo 'en\w+')
export NCCL_SOCKET_IFNAME=$socket_ifname

DATA_PATH=/local_datasets/Epickitchens100_clips/video
VMAE_PATH=/data/datasets/Epickitchens100_clips/epic_checkpoint-2400.pth
CLIP_PATH=/data/datasets/Epickitchens100_clips/ViT-B-16.pt
OUTPUT_DIR=$4 # weight저장.
MASTER_NODE=$1

python -u /data/psh68380/repos/ASD_capstone/main.py \
--data_root "/local_datasets/ASD/asd_ver2_all_5folds_annotation" \
--annotations_root "/data/psh68380/repos/ASD_capstone/part_proportion.csv" \
--num_epochs 50 \
--batch_size 4 \
--learning_rate 0.001 \
--image_model "efficientnetb0" \
--part_model "linear"

# OMP_NUM_THREADS=1 torchrun \
#     --nproc_per_node=$6 \
#     --master_port $3 --nnodes=$5 \
#     --node_rank=$2 --master_addr=${MASTER_NODE} \
#     /data/jong980812/project/cil/videoCIL/run_cil.py \
#     --model AIM_final \
#     --data_set SSV2 \
#     --data_path /local_datasets/something-something/something-something-v2-mp4 \
#     --anno_path /data/jong980812/project/cil/videoCIL/data/ssv2/annotation/ssv2_data_tasks_109_2.pkl \
#     --num_tasks 10 \
#     --log_dir  ${OUTPUT_DIR} \
#     --output_dir  ${OUTPUT_DIR} \
#     --batch_size 24 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224 \
#     --num_frames 8 \
#     --opt adamw \
#     --lr 5e-4 \
#     --opt_betas 0.9 0.999 \
#     --weight_decay 0.05 \
#     --epochs 50 \
#     --warmup_epochs 5 \
#     --dist_eval \
#     --memory_size 8000 \
#     --rehearsal_epochs 25 \
#     --init_scale 1.0 \
#     --num_workers 12 \
#     --dim_mlp 192 \
#     --mixup_prob 0 \
#     --mixup_switch_prob 0 \
#     --cutmix 0. \
#     --unfreeze_layers head decoder associator  \
#     --unfreeze_layers_after_base head decoder associator  \
#     --ba_layers 2 \
#     --temp_mode attention \
#     --uniform_ratio 2 \
#     --sampling_rate 3 \
#     --use_aim_weight /data/yuri1255/project/videoCIL/result/rehearsal_fix/ssv2/AIM_3480/OUT/checkpoint/task1_checkpoint.pth \
#     --get_frame_index \
#     --handcrafted_selection \
#     --rehearsal_samples_per_class 4 \
#     --fs_topk 4 \
#     --frame_matching \
#     --token_matching \
#     --replay_token \
#     --prompt_mode cross \
#     --len_prompt 8 \
#     --memory_mode global \

    
    # --ssv2_first_finetune /data/jong980812/project/cil/videoCIL/ssv2_4frame_novirtual_base.pth \
    

    # --use_aim_weight /data/yuri1255/project/videoCIL/result/rehearsal_fix/ssv2/AIM_3480/OUT/checkpoint/task1_checkpoint.pth \




    # --cos_temp 2

    # --no_training      
    # --unfreeze_layers head decoder temporal_embedding prompts \
    # --unfreeze_layers_after_base head decoder prompts \
    # --ba_layers 2 \
    # --temp_mode attention \
    # --

    
