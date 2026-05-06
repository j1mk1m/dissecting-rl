# test_nccl.py
import torch
import torch.distributed as dist
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

dist.init_process_group("nccl", rank=0, world_size=1)
t = torch.zeros(1).cuda()
print("NCCL init OK:", t)
dist.destroy_process_group()