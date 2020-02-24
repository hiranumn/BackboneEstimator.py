import os
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from scipy.spatial import distance, distance_matrix

class dataloader:
    
    def __init__(self,
                 proteins, # list of proteins to load'
                 atom = "CA",
                 datadir="/net/scratch/hiranumn/raw_relaxed2/", # Base directory for all protein data
                 lengthmax=500, # Limit to the length of proteins, if bigger we ignore.
                 load_dtype=np.float32, # Data type to load with
                 digitization1 = [-20.0, -15.0, -10.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 10.0, 15.0, 20.0],
                 verbose=False,
                 distance_cutoff = 4
                ):
        
        self.n = {}
        self.samples_dict = {}
        self.sizes = {}
        self.atom = atom
        
        self.load_dtype = load_dtype
        self.digitization1 = digitization1
        self.datadir = datadir
        self.verbose = verbose
        self.distance_cutoff = distance_cutoff
            
        # Loading file availability
        temp = []
        for p in proteins:
            path = join(datadir, p)
            samples_files = [f[:-4] for f in listdir(path) if isfile(join(path, f)) and ".pdb" in f]
            np.random.shuffle(samples_files)
            samples = samples_files
                    
            if len(samples) > 0:
                length = len(parsePDB(join(path, samples[0]+".pdb"))[1])
                if length < lengthmax:
                    temp.append(p)
                    self.samples_dict[p] = samples
                    self.n[p] = len(samples)
                    self.sizes[p] = length
                
        # Make a list of proteins
        self.proteins = temp
        self.index = np.arange(len(self.proteins))
        np.random.shuffle(self.index)
        self.cur_index = 0

    def next(self, transform=True, pindex=-1):
        pname = self.proteins[self.index[self.cur_index]]
        if pindex == -1:
            pindex = np.random.choice(np.arange(self.n[pname]))
        sample = self.samples_dict[pname][pindex]
        psize = self.sizes[pname]
        
        # Get coordinates
        coords, aas = parsePDB(join(self.datadir, pname, sample+".pdb"), atom=self.atom)
        native_coords = parsePDB(join(self.datadir, pname, "native.pdb"), atom=self.atom)[0]
        
        # Get maps
        maps = distance_matrix(coords, coords)
        native_maps = distance_matrix(native_coords, native_coords)
        sep = seqsep(psize)
        
        aas = [residuemap[i] for i in aas]
        
        # Get target
        estogram = get_estogram((maps, native_maps), self.digitization1)
        
        # Transform input distance
        if transform:
            maps = f(maps, cutoff=self.distance_cutoff)
        
        # Shuffle data if it gets to the end of protein list
        self.cur_index += 1
        if self.cur_index == len(self.proteins):        
            self.cur_index = 0 
            np.random.shuffle(self.index)
            
        if self.verbose:
            print(maps.shape, sep.shape)
            print(estogram.shape)
        
        output = np.concatenate([np.expand_dims(maps, axis=-1), sep], axis=-1), estogram, native_maps, np.eye(20)[aas]
        for i in output:
            assert np.sum(np.isnan(i)) == 0
        return output
    
def f(X, cutoff=6, scaling=3.0):
    X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
    return np.arcsinh(X_prime)/scaling

def get_estogram(XY, digitization):
    (X,Y) = XY
    residual = X-Y
    estogram = np.eye(len(digitization)+1)[np.digitize(residual, digitization)]
    return estogram

# Sequence separtion features
def seqsep(psize, normalizer=100, axis=-1):
    ret = np.ones((psize, psize))
    for i in range(psize):
        for j in range(psize):
            ret[i,j] = abs(i-j)*1.0/100-1.0
    return np.expand_dims(ret, axis)

def apply_label_smoothing(x, alpha=0.2, axis=-1):
    minind = 0
    maxind = x.shape[axis]-1
    
    # Index of true and semi-true labels
    index = np.argmax(x, axis=axis)
    lower = np.clip(index-1, minind, maxind)
    higher = np.clip(index+1, minind, maxind)
    
    # Location-aware label smoothing
    true = np.eye(maxind+1)[index]*(1-alpha)
    semi_lower = np.eye(maxind+1)[lower]*(alpha/2)
    semi_higher= np.eye(maxind+1)[higher]*(alpha/2)
    
    return true+semi_lower+semi_higher

def parsePDB(filename, atom="CA"):
    file = open(filename, "r")
    lines = file.readlines()
    coords = []
    aas = []
    
    cur_resdex = -1
    aa = ""
    for line in lines:
        if "ATOM" in line:
            if cur_resdex != int(line[22:26]):
                cur_resdex = int(line[22:26])
                new_res = True
                aa = line[17:20]
                aas.append(aa)
            if atom == "CA" and atom in line[12:16]:
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                coords.append(xyz)
            elif atom == "CB":
                if aa == "GLY" and "CA" in line[12:16]:
                    xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    coords.append(xyz)
                elif "CB" in line[12:16]:
                    xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    coords.append(xyz)
    return np.array(coords), aas

residues= ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',\
           'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',\
           'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
residuemap = dict([(residues[i], i) for i in range(len(residues))])