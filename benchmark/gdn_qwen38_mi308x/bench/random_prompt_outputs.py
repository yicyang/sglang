"""Send fixed random-token prompts (like bench_serving --dataset-name random) and
record greedy outputs, to compare token diversity across serving variants."""
import json
import sys

import requests
import torch

port, out = int(sys.argv[1]), sys.argv[2]
g = torch.Generator().manual_seed(0)
res = []
for i in range(16):
    ids = torch.randint(1000, 150000, (2000,), generator=g).tolist()
    r = requests.post(f"http://localhost:{port}/generate", json={
        "input_ids": ids, "sampling_params": {"max_new_tokens": 200, "temperature": 0, "ignore_eos": True},
        "return_logprob": False}).json()
    res.append(r["output_ids"] if "output_ids" in r else r["text"])
json.dump(res, open(out, "w"))
uniq = [len(set(x)) for x in res]
print("distinct output tokens per request:", uniq, "mean", sum(uniq) / len(uniq))
