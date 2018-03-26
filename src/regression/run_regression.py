#script for loading data and running regression

from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
from sys import argv
import pandas as pd
from regression_network import RegressionNetwork

def parse(args):
    '''
    Parses command line arguments.
    '''
    parser = ArgumentParser()
    parser.add_argument("--data", help="location of data npz file")
    parser.add_argument("--out", help="output location of trained model")
    return parser.parse_args(args)


def run(args):

    #load data from npz
    dat = np.load(args.data)


    #convert to pandas array
    cc_data = pd.DataFrame.from_records(dat['data'])

    images = dat['images']

    #select

    network = RegressionNetwork(images, cc_data['Array voltage V'].as_matrix(), args.out)

    network.train()



if __name__=="__main__":
    run(parse(argv[1:]))
