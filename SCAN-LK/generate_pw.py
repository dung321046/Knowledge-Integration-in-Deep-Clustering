"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os

import torch
from termcolor import colored

from utils.common_config import get_train_transformations, get_train_dataset
from utils.config import create_config

FLAGS = argparse.ArgumentParser(description='SCAN Loss')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')


def main():
    args = FLAGS.parse_args()
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))
    # CUDNN
    torch.backends.cudnn.benchmark = True
    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    train_dataset = get_train_dataset(p, train_transformations,
                                      split='train', to_neighbors_dataset=True, pw_dict=None)
    num_pw = 1000
    r = 0.5
    from utils.constraint_utils import generate_pw_constraints
    path = "C:\\Users\\dung3\\PycharmProjects\\Unsupervised-Classification\\constraints\\" + p[
        'train_db_name'] + "-pw" + str(num_pw) + "-" + str(r)
    os.makedirs(path)
    generate_pw_constraints(path, train_dataset.dataset.targets, num_pw, r=r)


if __name__ == "__main__":
    main()
