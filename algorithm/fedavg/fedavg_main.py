import os
import sys
import torch
import wandb
import numpy as np
import argparse
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from algorithm.fedavg.server import Server


def parse_args():
    # args default values
    config = {
        "model": 'mlp',
        "dataset": 'movielens',
        "client_num_in_total": 200,
        "client_num_per_round": 40,
        "num_rounds": 500,
        "partition_method": 'centralized',
        "client_optimizer": "adam",
        "lr": 0.003,
        "batch_size": 64,
        "epoch": 2,
        "eval_interval": 2,
        "seed": 42,
        "device": 'cuda',
        "lr_decay": 0.996,
        "decay_step": 20,
        "early_stop": 50,
        "wandb_mode": 'run',
        "notes": 'neg2pos_1_test',
    }

    config_mnist = {
        "model": 'cnn',
        "dataset": 'mnist',
        "client_num_in_total": 200,
        "client_num_per_round": 20,
        "num_rounds": 500,
        "partition_method": 'homo',
        "client_optimizer": 'sgd',
        "lr": 0.003,
        "batch_size": 32,
        "epoch": 2,
        "eval_interval": 1,
        "seed": 42,
        "device": 'cuda',
        "lr_decay": 0.998,
        "decay_step": 200,
        "wandb_mode": 'run',
        "notes": '',
    }

    parser = argparse.ArgumentParser(description='*******FedAvg Experiments Params*******')

    parser.add_argument('--model', type=str, default='cnn', metavar='N',
                        choices=['cnn', 'mlp', 'fm', 'lr', 'widedeep'],
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

    parser.add_argument('--early_stop', help='stop training if your model stops improving for early_stop rounds',
                        type=int, default=50)

    # use values from config dict by default
    parser.set_defaults(**config)

    # override with command line arguments when provided
    args = parser.parse_known_args()[0]

    return args


def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为单个GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # 消除cudnn卷积操作优化的精度损失


if __name__ == '__main__':
    args = parse_args()

    if args.partition_method == "centralized":
        args.client_num_in_total = 1
        args.client_num_per_round = 1

    assert args.client_num_in_total >= args.client_num_per_round, "choose too much clients per round"

    # Reproduction : select clients per round, dataloader shuffle, model parameter init...
    setup_seed(args.seed)

    wandb.init(project="sweep",
               name=str(args.partition_method)[:2].upper() + "-" + str(args.model)
                    + "-e_" + str(args.epoch)
                    + "-b_" + str(args.batch_size) + "-lr_" + str(args.lr) + "-"
                    + str(args.notes),
               tags=['CTR'],
               notes=args.notes,
               mode=args.wandb_mode,
               config=args)
    # Docs: https://docs.wandb.ai/library#logged-with-specific-calls

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
