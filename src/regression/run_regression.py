#script for loading data and running regression

from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
from sys import argv

def parse(args):
    '''
    Parses command line arguments.
    '''
    parser = ArgumentParser()
    parser.add_argument("--data", help="location of data CSV file")
    parser.add_argument("--out", help="output location of trained model")
    return parser.parse_args(args)


def run(args):


if __name__=="__main__":
    run(parse_args(argv[1:]))
