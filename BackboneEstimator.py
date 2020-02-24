import sys
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing
import time

def main():
    #####################
    # Parsing arguments
    #####################
    parser = argparse.ArgumentParser(description="Backbone Quality Estimator",
                                     epilog="v0.0.1")
    parser.add_argument("infolder",
                        action="store",
                        help="input folder name full of pdbs or path to a single pdb")
    
    parser.add_argument("outfolder",
                        action="store", nargs=argparse.REMAINDER,
                        help="output folder name. If a pdb path is passed this needs to be a .npz file. Can also be empty. Default is current folder or pdbname.npz")
    
    parser.add_argument("--pdb",
                        "-pdb",
                        action="store_true",
                        default=False,
                        help="Running on a single pdb file instead of a folder (Default: False)")
    
    parser.add_argument("--ca",
                        "-ca",
                        action="store_true",
                        default=False,
                        help="Predicting based on a calpha distance map (Default: False)")

    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        default=False,
                        help="verbose flag (Default: False)")
    
    parser.add_argument("--gpu",
                        "-g", action="store",
                        type=int,
                        default=0,
                        help="gpu device to use (default gpu0)")
    
    args = parser.parse_args()

    ################################
    # Checking file availabilities #
    ################################
    # made outfolder an optional positinal argument. So check manually it's lenght and unpack the string
    if len(args.outfolder)>1:
        print("Only one output folder can be specified, but got {args.outfolder}", file=sys.stderr)
        return -1

    if len(args.outfolder)==0:
        args.outfolder = ""
    else:
        args.outfolder = args.outfolder[0]

    if args.infolder.endswith('.pdb'):
        args.pdb = True
    
    if not args.pdb:
        if not isdir(args.infolder):
            print("Input folder does not exist.", file=sys.stderr)
            return -1
        
        #default is current folder
        if args.outfolder == "":
            args.outfolder='.'
        if not isdir(args.outfolder):
            print("Creating output folder:", args.outfolder)
            os.mkdir(args.outfolder)
    else:
        if not isfile(args.infolder):
            print("Input file does not exist.", file=sys.stderr)
            return -1
        
        #default is output name with extension changed to npz
        if args.outfolder == "":
            args.outfolder = os.path.splitext(args.infolder)[0]+".npz"

        if not(".pdb" in args.infolder and ".npz" in args.outfolder):
            print("Input needs to be in .pdb format, and output needs to be in .npz format.", file=sys.stderr)
            return -1
            
    
    script_dir = os.path.dirname(__file__)
    base = os.path.join(script_dir, "models/")
    
    ##############################
    # Importing larger libraries #
    ##############################
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    script_dir = os.path.dirname(__file__)
    sys.path.insert(0, script_dir)
    import pyErrorPred
    
    if args.ca:
        m = pyErrorPred.Model(name=join(base, "CA-net"))
    else: 
        m = pyErrorPred.Model(name=join(base, "CB-net"))
    m.load()
        
    #########################
    # Getting samples names #
    #########################
    if not args.pdb:
        ###########################
        # Prediction happens here #
        ###########################                                  
        samples = [f for f in listdir(args.infolder) if f.endswith(".pdb")]
        for s in samples:
            start_time = time.time()
            if args.ca:
                temp = pyErrorPred.getData(join(args.infolder, s), "CA")
            else:
                temp = pyErrorPred.getData(join(args.infolder, s), "CB")
            lddt, estogram, mask = m.predict2((temp, "dmy", "dmy", "dmy"))
            np.savez_compressed(join(args.outfolder, s[:-4]+".npz"),
                                lddt = lddt,
                                estogram = estogram,
                                mask = mask)
            if args.verbose: print("Processed "+s+" (%0.2f seconds)" % (time.time() - start_time))

    # Processing for single sample
    else:
        infilepath = args.infolder
        outfilepath = args.outfolder
        infolder = "/".join(infilepath.split("/")[:-1])
        insamplename = infilepath.split("/")[-1][:-4]
        outfolder = "/".join(outfilepath.split("/")[:-1])
        outsamplename = outfilepath.split("/")[-1][:-4]
        
        if args.verbose: 
            print("Only working on a single file:", outfolder, outsamplename)
                              
        sample = insamplename+".pdb"
        if args.ca:
            temp = pyErrorPred.getData(join(infolder, sample), "CA")
        else:
            temp = pyErrorPred.getData(join(infolder, sample), "CB")

        lddt, estogram, mask = m.predict2((temp, "dmy", "dmy", "dmy"))
                              
        np.savez_compressed(join(outfolder, outsamplename+".npz"),
                            lddt = lddt,
                            estogram = estogram,
                            mask = mask)
                              
if __name__== "__main__":
    main()
