#!/bin/bash
#SBATCH --job-name=HLLM-train-repo
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2

#SBATCH --partition=general  
#SBATCH --mem=256G 
#SBATCH --gres=gpu:A100_80GB:2

#SBATCH --nodelist=babel-5-[23,27,31]

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

echo "Job Starts"

eval "$(conda shell.bash hook)"
conda activate hllm

echo "activated"

user_pretrain_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-intermediate-step-1431k-3T"
item_pretrain_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-intermediate-step-1431k-3T"
# item_pretrain_dir="/data/user_data/jingyuah/LLARA/checkpoints/pixel_200k_item_token_EBAE_logs/checkpoint-1400/Pixel200K_SFT_fixed_tr0.2_lr_1e-6/checkpoint-1500/causal_lm_dir"
# item_pretrain_dir="/data/user_data/jingyuah/LLARA/checkpoints/pixel_200k_item_token_EBAE_logs/checkpoint-1400"

# item_emb_pretrain="${item_pretrain_dir}/item_token_word_embed.bin"
checkpoint_dir="/data/group_data/cx_group/REC/checkpoints/HLLM-ml-1m"

item_emb_pretrain=""

checkpoint_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/repro_tiny_llama_1.1b_pixelrec_200K"

inter_path="/data/user_data/jingyuah/HLLM_weights/data/dataset"
info_path="/data/user_data/jingyuah/HLLM_weights/data/information"

epoch=5

auto_resume="${checkpoint_dir}/HLLM-0.pth" # --auto_resume  $auto_resume \

# Item and User LLM are initialized by specific pretrain_dir.
python3 /home/jingyuah/HLLM/code/main.py \
    --config_file /home/jingyuah/HLLM/code/overall/LLM_deepspeed.yaml /home/jingyuah/HLLM/code/HLLM/HLLM.yaml \
    --loss nce \
    --epochs $epoch \
    --dataset Pixel200K \
    --train_batch_size 8 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir $item_pretrain_dir \
    --user_pretrain_dir $user_pretrain_dir \
    --text_path $info_path \
    --data_path $inter_path \
    --text_keys '[\"title\",\"tag\",\"description\"]' # Please remove tag in books dataset.


# --item_emb_pretrain $item_emb_pretrain \

echo "Job Ends"
