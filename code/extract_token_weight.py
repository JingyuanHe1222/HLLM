import os 

import torch

from transformers import AutoModel

path = "/data/user_data/jingyuah/LLARA/checkpoints/pixel_200k_item_token_EBAE_logs/checkpoint-1400/Pixel200K_SFT_fixed_tr0.2_lr_1e-6/checkpoint-1500/causal_lm_dir"
model = AutoModel.from_pretrained(path, local_files_only=True)

token_weights = model.embed_tokens.weight

item_token_w = token_weights[0, :][None, None, :]

torch.save(item_token_w, os.path.join(path, "item_token_word_embed.bin"))
