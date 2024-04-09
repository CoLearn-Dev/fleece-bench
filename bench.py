import torch
import pandas as pd
import argparse

fleece_worker = __import__("fleece-worker")

worker = fleece_worker.Worker()

worker.start_layer_forward_engine()

parser = argparse.ArgumentParser(description='Run the estimation')
parser.add_argument('--model', '-m', type=str, required=True)
args = parser.parse_args()
model = args.model
df_layers = pd.read_csv("./specs/fleece_layers.csv")
layer_names = []
for idx, row in df_layers.iterrows():
    if not row["From_model"] == model:
        continue
    layer_name = row["Layer_name"]
    if row["Repetition"] == 1:
        layer_names.append(layer_name)
    else:
        for i in range(min(row["Repetition"], 5)):
            layer_names.append(f"{layer_name}.{i}")
# example 1
print("[")
worker.preload_layers(layer_names)
h = torch.tensor([[1, 518, 25580, 29962, 825, 338, 278, 9522, 412, 310, 1122, 11586, 895, 29973, 518, 29914, 25580, 29962]], device="cuda")
start_pos = 0
is_new_task = start_pos == 0
kv_cache_dict = dict()
for _ in range(16):
    bsz = h.shape[0]
    seqlen = h.shape[1]
    _, kv_cache_dict = worker.layers_forward(h, layer_names, bsz, is_new_task, 0, start_pos, seqlen, kv_cache_dict)
    is_new_task = False
    start_pos += seqlen
    h = torch.tensor([[29962]], device="cuda")

# # example 2
hidden_dim = 4092 if model == "llama-2-7b-chat-slice" else 8192 if model == "llama-2-70b-chat-slice" else 8192
layer_names = [f"{model}/layers.0", f"{model}/layers.1"]
worker.preload_layers(layer_names)
h = torch.randn((1, 18, 8192), dtype=torch.float16, device="cuda")
start_pos = 0
is_new_task = start_pos == 0
kv_cache_dict = dict()
for _ in range(16):
    bsz = h.shape[0]
    seqlen = h.shape[1]
    _, kv_cache_dict = worker.layers_forward(h, layer_names, bsz, is_new_task, 0, start_pos, seqlen, kv_cache_dict)
    is_new_task = False
    start_pos += seqlen
    h = torch.randn((1, 1, 8192), dtype=torch.float16, device="cuda")

print("]")