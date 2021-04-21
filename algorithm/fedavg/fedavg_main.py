# FedAvg Algorithm
import os
import sys
import torch
import wandb
import numpy as np
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))  # add project to PYTHONPATH

from algorithm.fedavg.server import Server


def parse_args():
    parser = argparse.ArgumentParser(description='*******FedAvg Experiments Params*******')

    parser.add_argument('--model', type=str, default='cnn_mnist', metavar='N',
                        choices=['cnn_mnist', 'lr_ctr', 'dnn_ctr'],
                        help='model used for training')

    parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                        help='dataset used for training')

    parser.add_argument('--client_num_in_total', type=int, default=200, metavar='NN',
                        help='number of clients in distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of clients selected per round')

    parser.add_argument('--num_rounds', type=int, default=100, metavar='NR',
                        help='how many round of communications we should use')

    parser.add_argument('--partition_method', type=str, default='centralized', metavar='N',
                        choices=['hetero', 'homo', 'centralized'],
                        help='how to partition the dataset on local workers')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='your model optimizer')

    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate for local optimizers (default: 3e-4)')

    parser.add_argument('--batch_size', type=int, default=32, metavar='BS',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--epoch', type=int, default=2, metavar='E',
                        help='how many epochs will be trained locally')

    parser.add_argument('--eval_interval', type=int, default=2, metavar='EV',
                        help='the interval communication rounds to do an evaluation on all train/test datasets')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='seed for random select client')

    parser.add_argument('--device', type=str, default='cuda', metavar='DV',
                        choices=['cuda', 'cpu'],
                        help='the device to your training (default: cuda)')

    parser.add_argument('--wandb_mode', type=str, default='run', metavar='WM',
                        choices=['run', 'disabled', 'offline'],
                        help='Whether to use wandb to visualize experimental results (run, disabled, offline)')

    parser.add_argument('--notes', type=str, default='', metavar='NT',
                        help='wandb remark information')

    parser.add_argument('--lr_decay', help='sgd: decay rate for learning rate', type=float, default=0.998)

    parser.add_argument('--decay_step', help='sgd: decay step for learning rate', type=int, default=200)

    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.partition_method == "centralized":
        args.client_num_in_total = 1
        args.client_num_per_round = 1

    # 使模型随机性+dataloader shuffle不具有随机性
    # 随机选择客户端的随机性要动态的
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 初始化wandb
    wandb.init(project="FedPro",
               name=str(args.dataset) + "_" + str(args.partition_method) + "_" + str(args.notes),  # 这个是图表的名称
               tags=['GPU', 'working'],
               notes=args.notes,  # https://docs.wandb.ai/library#logged-with-specific-calls
               mode=args.wandb_mode,
               config=args)

    print(f"############## Running FedAvg With ##############\n"
          f"algorithm:\t\t\t\t\tfedavg\n"
          f"dataset:\t\t\t\t\t{args.dataset}\n"
          f"model:\t\t\t\t\t\t{args.model}\n"
          f"num_rounds:\t\t\t\t\t{args.num_rounds}\n"
          f"client_num_in_total:\t\t{args.client_num_in_total}\n"
          f"client_num_per_round:\t\t{args.client_num_per_round}\n"
          f"partition_method:\t\t\t{args.partition_method}\n"
          f"eval_interval:\t\t\t\t{args.eval_interval}\n"
          f"batch_size:\t\t\t\t\t{args.batch_size}\n"
          f"epoch:\t\t\t\t\t\t{args.epoch}\n"
          f"lr:\t\t\t\t\t\t\t{args.lr}\n"
          f"optimizer:\t\t\t\t\t{args.client_optimizer}\n"
          f"device:\t\t\t\t\t\t{args.device}\n"
          f"##################################################\n")

    server = Server(args)

    server.federate()
