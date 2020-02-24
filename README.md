# BackboneEstimator.py
Estimates backbone quality based on C beta (or C alpha, optionally) distance maps.

```
usage: BackboneEstimator.py [-h] [--pdb] [--ca] [--verbose] [--gpu GPU] infolder ...

Backbone Quality Estimator

positional arguments:
  infolder           input folder name full of pdbs or path to a single pdb
  outfolder          output folder name. If a pdb path is passed this needs to be a .npz file. Can also be empty. Default
                     is current folder or pdbname.npz

optional arguments:
  -h, --help         show this help message and exit
  --pdb, -pdb        Running on a single pdb file instead of a folder (Default: False)
  --ca, -ca          Predicting based on a calpha distance map (Default: False)
  --verbose, -v      verbose flag (Default: False)
  --gpu GPU, -g GPU  gpu device to use (default gpu0)

v0.0.1
```
# Example usages (for IPD people)
Type the following commands to activate tensorflow environment with pyrosetta3.
```
source activate tensorflow
source /software/pyrosetta3/setup.sh
```

Running on a folder of pdbs with Cbeta coordinates  (foldername: ```samples```)
```
python BackboneEstimator.py -v samples out
```

Running on a folder of pdbs with Calpha coordinates (foldername: ```samples```)
```
python BackboneEstimator.py -v -ca samples out
```

Running on a single pdb file (inputname: ```input.pdb```). Output name is optional and defaults to input.npz
```
python BackboneEstimator.py -v --pdb input.pdb [output.npz]
```

# How to look at outputs
Output of the network is written to ```[input_file_name].npz.```
You can extract the predictions as follows.

```
import numpy as np

x = np.load("testoutput.npz")

lddt = x["lddt"]           # per residue lddt
estogram = x["estogram"]   # per pairwise distance e-stogram
mask = x["mask"]           # mask predicting native < 15
```
Perhaps ```lddt``` is the easiest place to start as it is per-residue quality score. You can simply take an average if you want a global score per protein structure. 

If you want to do something more involved, especially for protein complex design, see [example.ipynb](ipynbs/example.ipynb) for getting more specialized metrics. If you want to play with pair-wise error predictions, [samples.ipynb](ipynbs/samples.ipynb) is a good place to start.

# Required softwares
- Python3.5>
- Pyrosetta 
- Tensorflow 1.14 (not Tensorflow 2.0)
