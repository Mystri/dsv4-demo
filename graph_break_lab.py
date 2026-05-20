import torch
import os

os.environ["TORCH_LOGS"] = "graph_breaks"

@torch.compile
def break_if(x):
    if x.sum() > 0:      # 隐式 .item()
        return x * 2
    return x

break_if(torch.randn(3))

torch._dynamo.explain(fn)(x)
