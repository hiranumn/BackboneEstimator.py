import sys
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing

import sys
sys.path.insert(0, "./")
import pyErrorPred as pye

def main():
    parser = argparse.ArgumentParser(description="Error predictor network trainer",
                                     epilog="v0.0.1")
    
    parser.add_argument("folder",
                        action="store",
                        help="Location of folder to save checkpoints to.")
    
    parser.add_argument("--epoch",
                        "-e", action="store",
                        type=int,
                        default=200,
                        help="# of epochs (path over all proteins) to train for (Default: 200)")
    
    parser.add_argument("--useCA",
                        "-ca",
                        action="store_true",
                        default=False,
                        help="using CA coords instead of CB (Default: False)")
    
    parser.add_argument("--decay",
                        "-d", action="store",
                        type=float,
                        default=0.99,
                        help="Decay rate for learning rate (Default: 0.99)")
    
    parser.add_argument("--base",
                        "-b", action="store",
                        type=float,
                        default=0.0005,
                        help="Base learning rate (Default: 0.0005)")
    
    parser.add_argument("--silent",
                        "-s",
                        action="store_true",
                        default=False,
                        help="Run in silent mode (Default: False)")
   
    args = parser.parse_args()
    
    restoreModel = False
    if isdir(args.folder):
        restoreModel = True
    
        
    if not args.silent:
        print("Loading samples")
    ##########################
    ### Loading data files ###
    ##########################
    script_dir = os.path.dirname(__file__)
    base = join(script_dir, "data/")
    
    X = pye.dataloader(np.load(join(base,"train_proteins.npy")),
                       lengthmax = 280,
                       atom = "CA" if args.useCA else "CB")
    V = pye.dataloader(np.load(join(base,"valid_proteins.npy")),
                       lengthmax = 280,
                       atom = "CA" if args.useCA else "CB")
    
    if not args.silent:
        print("Building a network")
    #########################
    ### Training a model  ###
    #########################
    m = pye.Model(name=args.folder)
    
    if restoreModel:
        model.load()
   
    if not args.silent:
        print("Training the network")
        
    m.train(X,
            V,
            args.epoch,
            decay=args.decay,
            base_learning_rate=args.base,
            save_best=True)
    
    return 0

if __name__== "__main__":
    main()
        