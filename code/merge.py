import numpy as np
import os 
import sys
import pickle 

from tqdm import tqdm 


def write_embed_to_binary(embeddings, output_path): 
    """
    Write the embedding array into a binary file in ANN-Indexing (DiskANN, SPTAG) format. 
    The content of the output file can be access through: embeds = read_fbin(output_path)
    """
    num, dim = embeddings.shape
    with open(output_path, "wb") as f:
        f.write(num.to_bytes(4, 'little'))
        f.write(dim.to_bytes(4, 'little'))
        f.write(embeddings.tobytes())




base_dir = "/data/group_data/cx_group/REC/ClueWeb-Reco/HLLM_exps"
emb_map = {
    "HLLM_ml-1m": (0, 7, 16), 
    "HLLM_item_encoding_ml-1m": (32, 63, 64)
}
output_path = "/data/group_data/cx_group/REC/ClueWeb-Reco/HLLM_exps/item_embed_full.bin"

all_embeds = []
for folder_ in ["HLLM_ml-1m", "HLLM_item_encoding_ml-1m"]: 
    folder_path = os.path.join(base_dir, folder_)
    start, end, num_shards = emb_map[folder_]
    for i in tqdm(range(start, end+1)): 
        file_path = os.path.join(folder_path, f"clueweb-b-en.{i}-of-{num_shards}.pkl")
        # output 
        print(f"Opening {file_path}")
        sys.stdout.flush()
        # load item features 
        with open(file_path, "rb") as f:
            embeds = pickle.load(f)
        all_embeds.append(embeds)
        del embeds 

all_embeds = np.concatenate(all_embeds, 0)

print("final embed shape: ", all_embeds.shape)

write_embed_to_binary(all_embeds, output_path)
