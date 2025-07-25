import numpy as np
import os
import pickle 
import torch 



def read_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        print("number of queries: ", nvecs)
        print("dimension: ", dim)
        f.seek(4+4)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def read_pickle_embed(filename): 
    with open(filename, 'rb') as f:
        embed = pickle.load(f)
    return embed 


base_dir="/data/user_data/jingyuah/HLLM"


# ############## sequence: first batch ##################
hllm_seq_emb_path = os.path.join(base_dir, 'first_seq_emb.pkl')
seq_emb_path = os.path.join(base_dir, 'seq_embed.bin')

# official seq emb 
hllm_emb = read_pickle_embed(hllm_seq_emb_path).cpu()
hllm_emb = hllm_emb / hllm_emb.norm(dim=-1, keepdim=True)
hllm_emb_1 = hllm_emb[:128, :]

# seq encoding script seq emb 
seq_emb = read_fbin(seq_emb_path)
seq_emb = torch.from_numpy(seq_emb).to(torch.bfloat16)
seq_emb = seq_emb / seq_emb.norm(dim=-1, keepdim=True)
seq_emb_1 = seq_emb[:128, :]

print(torch.allclose(hllm_emb_1, seq_emb_1, atol=1e-16))



# ############## item features ##################
hllm_item_emb_path = os.path.join(base_dir, 'ml.pkl')
item_emb_path = os.path.join(base_dir, 'item_embed_full_streamed.bin')

hllm_item_emb = read_pickle_embed(hllm_item_emb_path)
hllm_item_emb = torch.from_numpy(hllm_item_emb).to(torch.bfloat16)
hllm_item_emb = hllm_item_emb / hllm_item_emb.norm(dim=-1, keepdim=True)

item_emb = read_fbin(item_emb_path)
item_emb = torch.from_numpy(item_emb).to(torch.bfloat16)
item_emb = item_emb / item_emb.norm(dim=-1, keepdim=True)

print(torch.allclose(hllm_item_emb, item_emb, atol=1e-16))



# ################## scores ################# 
hllm_scores = torch.matmul(hllm_emb_1.to("cuda"), hllm_item_emb.to("cuda").t()) 
scores = torch.matmul(seq_emb_1.to("cuda"), item_emb.to("cuda").t()) 

breakpoint()

print(torch.allclose(hllm_scores, scores, atol=1e-16))