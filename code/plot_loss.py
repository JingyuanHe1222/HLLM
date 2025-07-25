import os
import numpy as np
from matplotlib import pyplot as plt 
import seaborn as sns

DEFAULT_ALPHA = 0.4

def get_group_color_scheme(num_colors):

    palette = sns.color_palette("husl", num_colors)
    return palette[:num_colors]



selected_plot_metrics = ["loss", "nce_top10_acc", "nce_top1_acc", "nce_top100_acc"]
expt_name = "HLLM-train-item-only-2237394" # 290283
font_size = 12
label_size = 8
width_ = 12
height_ = 16
period = 100
span = 1

# the reproduced baseline result 
metrics_base = {
            "lr": [], "loss": [], "data": [], "fwd": [], "bwd": [], "nce_samples": [], "nce_top1_acc": [],
            "nce_top5_acc": [], "nce_top10_acc": [], "nce_top50_acc": [], "nce_top100_acc": []
        }
repo_log_out_file = "/home/jingyuah/HLLM/code/outputs/HLLM-train-repro.out"
with open(repo_log_out_file, 'r') as f:
    for line in f: 
        if "loss: " in line: 
            processed_line = "lr:" + line.strip().split('lr:')[-1]
            if "epoch" in processed_line: # epoch stats
                continue
            processed_line = processed_line.split(']')[0]
            processed_line = processed_line.replace(':', '')
            parts = processed_line.split(' ')
            idx = 0
            while (idx < len(parts)): 
                if parts[idx] == 'lr30': 
                    breakpoint()
                if len(parts[idx]) == 0 or parts[idx] == " ":
                    idx += 1
                    continue 
                metrics_base[parts[idx]].append(float(parts[idx+1]))
                idx += 2

metrics_exp = {
            "lr": [], "loss": [], "data": [], "fwd": [], "bwd": [], "nce_samples": [], "nce_top1_acc": [],
            "nce_top5_acc": [], "nce_top10_acc": [], "nce_top50_acc": [], "nce_top100_acc": []
        }
repo_log_out_file = f"/home/jingyuah/HLLM/code/outputs/{expt_name}.out"
with open(repo_log_out_file, 'r') as f:
    for line in f: 
        if "INFO  lr:" in line: 
            processed_line = "lr:" + line.strip().split('lr:')[-1]
            if "epoch" in processed_line: # epoch stats
                continue
            processed_line = processed_line.split(']')[0]
            processed_line = processed_line.replace(':', '')
            parts = processed_line.split(' ')
            idx = 0
            while (idx < len(parts)): 
                if parts[idx] == 'lr06': 
                    continue
                if len(parts[idx]) == 0 or parts[idx] == " ":
                    idx += 1
                    continue 
                metrics_exp[parts[idx]].append(float(parts[idx+1]))
                idx += 2


colors = get_group_color_scheme(2)

valid_step = len(metrics_base['loss'][:period]) // span
index_span = np.arange(valid_step, dtype=int)
index_span = index_span * span

if not os.path.isdir(f'outputs/{expt_name}'): 
    os.mkdir(f'outputs/{expt_name}')

for i in range(len(selected_plot_metrics)): 
    metric = selected_plot_metrics[i]
    plt.plot(np.array(metrics_base[metric][:period])[index_span], color=colors[0], label="HLLM base")
    plt.plot(np.array(metrics_exp[metric][:period])[index_span], color=colors[1], label=expt_name)

    plt.title(f"{metric} vs. step (20)")
    plt.xlabel('step (20)', fontsize=font_size)

    # plt.title("Recall@100 vs. Ratio of clusters searched")
    # plt.xlabel('Ratio of clusters searched', fontsize=font_size)

    plt.ylabel(metric, fontsize=font_size)
    plt.gcf().set_size_inches(width_, height_)
    plt.legend(prop={'size': label_size}, loc='lower right')
    plt.savefig(f'outputs/{expt_name}/{metric}.png', bbox_inches='tight')
    plt.close()
