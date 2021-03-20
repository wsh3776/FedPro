import argparse

# Optional Datasets
DATASETS = ['mnist', 'femnist', 'cifar10', 'cifar100']
# Optional Algorithms
MODELS = ['fedavg', 'fedprox']


def parse_args():
    parser = argparse.ArgumentParser(description='[Federated Learning Experiments]')

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

    parser.add_argument('--batch_size', type=int, default=8, metavar='BS',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--epoch', type=int, default=2, metavar='E',
                        help='how many epochs will be trained locally')

    parser.add_argument('--eval_interval', type=int, default=1, metavar='EV',
                        help='the interval communication rounds to do an evaluation on all train/test datasets')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='seed for random select client')

    parser.add_argument('--note', type=str, default='smile', metavar='NT',
                        help='remark information')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
