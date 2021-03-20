import os
import sys
import torch
import numpy as np

# Add your projects to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
# print(sys.path)

from utils.args import parse_args
from algorithm.fedavg.server import Server

if __name__ == '__main__':
    # get parameters
    args = parse_args()
    model = args.model
    dataset = args.dataset
    client_num_in_total = args.client_num_in_total
    client_num_per_round = args.client_num_per_round
    num_rounds = args.num_rounds
    lr = args.lr
    batch_size = args.batch_size
    epoch = args.epoch
    eval_interval = args.eval_interval
    seed = args.seed
    note = args.note

    # 使模型随机性+dataloader shuffle不具有随机性
    # 随机选择客户端的随机性要动态的
    np.random.seed(seed)
    torch.manual_seed(seed)  # recurrence experiment
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"############## Running FedAvg With ##############\n"
          f"algorithm:\t\t\t\t\tfedavg\n"
          f"model:\t\t\t\t\t\t{model}\n"
          f"dataset:\t\t\t\t\t{dataset}\n"
          f"num_rounds:\t\t\t\t\t{num_rounds}\n"
          f"client_num_in_total:\t\t{client_num_in_total}\n"
          f"client_num_per_round:\t\t{client_num_per_round}\n"
          f"batch_size:\t\t\t\t\t{batch_size}\n"
          f"epoch:\t\t\t\t\t\t{epoch}\n"
          f"lr:\t\t\t\t\t\t\t{lr}\n"
          f"seed:\t\t\t\t\t\t{seed}\n"
          f"note:\t\t\t\t\t\t{note}\n"
          f"##################################################\n"
          )

    server = Server(args)

    # TODO: this is the core api
    server.federate()

