#!/bin/bash
#SBATCH --job-name=HLLM-train-bge-m3_tinyllama
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2

#SBATCH --exclude=babel-3-21,babel-1-[23,27,31],babel-0-[19,23,27,31]

#SBATCH --partition=general  
#SBATCH --mem=256G 

#SBATCH --gres=gpu:A6000:4

#SBATCH --time=48:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user="jingyuah@cs.cmu.edu"

echo "Job Starts"

eval "$(conda shell.bash hook)"
conda activate hllm

echo "activated"

user_pretrain_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/TinyLlama-1.1B-intermediate-step-1431k-3T"
item_pretrain_dir="BAAI/bge-m3" 
# Mistral 4096: BAAI/bge-en-icl 
# Gemma2Model 3584: BAAI/bge-multilingual-gemma2
# XLMRobertaModel 1024: BAAI/bge-m3 
# BERTModel 768: BAAI/bge-base-en-v1.5

checkpoint_dir="/data/user_data/jingyuah/HLLM_weights/checkpoints/bge-m3_tinyllama1.1b_pixel200K"

inter_path="/data/user_data/jingyuah/HLLM_weights/data/dataset"
info_path="/data/user_data/jingyuah/HLLM_weights/data/information"

# no embed token 
item_emb_token_n=0

epoch=5
bz=16
stage=3 # for mistral + tinyllama

# Item and User LLM are initialized by specific pretrain_dir.
python3 /home/jingyuah/HLLM/code/main.py \
    --config_file /home/jingyuah/HLLM/code/overall/LLM_deepspeed.yaml /home/jingyuah/HLLM/code/HLLM/BGEHLLM.yaml \
    --loss nce \
    --epochs $epoch \
    --dataset Pixel200K \
    --item_emb_token_n $item_emb_token_n \
    --train_batch_size $bz \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --checkpoint_dir $checkpoint_dir \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir $item_pretrain_dir \
    --user_pretrain_dir $user_pretrain_dir \
    --text_path $info_path \
    --data_path $inter_path \
    --log_wandb True \
    --stage $stage \
    --text_keys '[\"title\",\"tag\",\"description\"]' # Please remove tag in books dataset.


# --item_emb_pretrain $item_emb_pretrain \

echo "Job Ends"
