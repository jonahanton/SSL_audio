"""Simple script to test distributed communication between cluster gpus"""
import torch
import torch.distributed as dist
import os
import json

rank = int(os.environ['RANK'])
gpu = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
env_dict = {
    key: os.environ[key]
    for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
}
print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

# dump environment variables to a json file 
if rank == 0:
    with open('environment_variables.json', mode='w') as jsonfile:
        json.dump(dict(os.environ), jsonfile, indent=2)


dist.init_process_group(backend='nccl')
print(
    f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
    + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
)
dist.barrier()