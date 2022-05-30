"""
A script to run multinode training on a SLURM cluster with submitit.
Almost copy-paste from 
    https://github.com/facebookresearch/dino/blob/main/run_with_submitit.py
    https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
"""
import argparse
import os
import uuid 
from pathlib import Path
import os

import submitit 
import main_pretrain

def parse_args():

    parser = argparse.ArgumentParser("Submitit for BT-Audio", parents=[main_pretrain.get_args_parser()])
    parser.add_argument("--ngpus", default=2, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--output_dir", default="", type=str)
    return parser.parse_args()


def get_shared_folder():
    if Path("/vol/bitbucket/jla21/proj/slurm/checkpoint/").is_dir():
        p = Path(f"/vol/bitbucket/jla21/proj/slurm/checkpoint/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.args.dist_url = get_init_file().as_uri()

    def __call__(self):
        self._setup_gpu_args()
        main_pretrain.train_bt(self.args)

    def checkpoint(self):
        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.output_dir == "":
        args.output_dir = get_shared_folder() / "%j"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.output_dir)

    executor.update_parameters(
        gpus_per_node=args.ngpus,
        tasks_per_node=args.ngpus, 
        nodes=args.nodes,
    )
    executor.update_parameters(name="bt-audio")

    args.dist_url = get_init_file().as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {args.output_dir}")


if __name__ == "__main__":
    main()
