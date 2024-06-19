import argparse
import random


def subsample_list(inp_list: list, sample_rate: float):
    random.shuffle(inp_list)
    return [inp_list[i] for i in range(int(len(inp_list) * sample_rate))]


def str2bool(v: str):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
