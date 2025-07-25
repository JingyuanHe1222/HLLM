#!/bin/bash
#SBATCH --job-name=HLLM-eval
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

#SBATCH --partition=general  
#SBATCH --mem=256G 
#SBATCH --gres=gpu:A100_80GB:1

#SBATCH --nodelist=babel-5-[23,27,31]

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

echo "Job Starts"

eval "$(conda shell.bash hook)"
conda activate hllm

echo "activated"

# user_pretrain_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-intermediate-step-1431k-3T"


user_pretrain_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-Chat-v0.4"
item_pretrain_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-Chat-v0.4"

# checkpoint_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/repro_tiny_llama_1.1b_pixelrec_200K"
checkpoint_dir="/data/group_data/cx_group/REC/checkpoints/HLLM-ml-1m"


dataset="Pixel200K"
inter_path="/data/user_data/jingyuah/HLLM_weights/data/dataset"
info_path="/data/user_data/jingyuah/HLLM_weights/data/information"

# dataset="ml1m"
# inter_path="/home/jingyuah/REC_source/HLLM/dataset"
# info_path="/home/jingyuah/REC_source/HLLM/information"

# Item and User LLM are initialized by specific pretrain_dir.
python3 /home/jingyuah/HLLM/code/main.py \
    --config_file /home/jingyuah/HLLM/code/overall/LLM_deepspeed.yaml /home/jingyuah/HLLM/code/HLLM/HLLM.yaml \
    --loss nce \
    --epochs 5 \
    --dataset $dataset \
    --train_batch_size 8 \
    --eval_batch_size 128 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir $item_pretrain_dir \
    --user_pretrain_dir $user_pretrain_dir \
    --text_path $info_path \
    --data_path $inter_path \
    --text_keys '[\"title\",\"tag\",\"description\"]' \
    --val_only True # Add this for evaluation

echo "Job Ends"