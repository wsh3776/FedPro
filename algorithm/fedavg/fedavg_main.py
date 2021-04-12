import os
import sys
import torch
import wandb
import numpy as np
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))  # add project to PYTHONPATH
from algorithm.fedavg.server import Server


def parse_args():
    parser = argparse.ArgumentParser(description='*******FedAvg Experiments Params*******')

    parser.add_argument('--model', type=str, default='cnn', metavar='N',
                        help='model used for training')

    parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                        help='dataset used for training')

    parser.add_argument('--client_num_in_total', type=int, default=200, metavar='NN',
                        help='number of clients in distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of clients selected per round')

    parser.add_argument('--num_rounds', type=int, default=100, metavar='NR',
                        help='how many round of communications we should use')

    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate for local optimizers (default: 3e-4)')

    parser.add_argument('--batch_size', type=int, default=32, metavar='BS',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--epoch', type=int, default=2, metavar='E',
                        help='how many epochs will be trained locally')

    parser.add_argument('--eval_interval', type=int, default=1, metavar='EV',
                        help='the interval communication rounds to do an evaluation on all train/test datasets')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='seed for random select client')

    parser.add_argument('--device', type=str, default='cuda', metavar='DV',
                        help='the device to your training (default: cuda)')

    parser.add_argument('--note', type=str, default='smile', metavar='NT',
                        help='remark information')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get parameters
    args = parse_args()

    if torch.cuda.is_available() == 0:
        assert args.device == 'cuda'

    # 使模型随机性+dataloader shuffle不具有随机性
    # 随机选择客户端的随机性要动态的
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # recurrence experiment
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 初始化wandb
    wandb.init(project="FedPro_test", name="demo", mode="run")  # run, disabled

    print(f"############## Running FedAvg With ##############\n"
          f"algorithm:\t\t\t\t\tfedavg\n"
          f"dataset:\t\t\t\t\t{args.dataset}\n"
          f"model:\t\t\t\t\t\t{args.model}\n"
          f"num_rounds:\t\t\t\t\t{args.num_rounds}\n"
          f"client_num_in_total:\t\t\t\t{args.client_num_in_total}\n"
          f"client_num_per_round:\t\t\t\t{args.client_num_per_round}\n"
          f"batch_size:\t\t\t\t\t{args.batch_size}\n"
          f"epoch:\t\t\t\t\t\t{args.epoch}\n"
          f"lr:\t\t\t\t\t\t{args.lr}\n"
          f"##################################################\n"
          )

    server = Server(args)

    # TODO: this is the core api
    server.federate()
