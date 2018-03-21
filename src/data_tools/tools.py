#loading, reformatting, and saving data

from sys import argv
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse(args):
    parser = ArgumentParser()
    parser.add_argument("--data", help="CSV file with timestamped data")
    return parser.parse_args(args)

def load_data(datafile):
    '''
    loads data from CSV and linked image arrays
    '''

    df = pd.read_csv(datafile)

    img_loc = os.path.dirname(datafile)

    files = df['filename']
    #NOTE: Image shape: (480, 640, 4)
    images = np.stack([np.fromfile("{}/{}".format(img_loc, file), dtype=np.uint8).reshape((480, 640, 4)) for file in files], axis=-1)

    return df, images

def run(args):
    '''
    debug methods for data loading
    '''
    dat, imgs = load_data(args.data)

    recs = dat.to_records()

    #saving data as npz
    np.savez_compressed("data_3_20.npz", data=recs, images=imgs)



if __name__=="__main__":
    run(parse(argv[1:]))
