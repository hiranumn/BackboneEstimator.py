import numpy as np
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import time
from .resnet import *
from .model import *
from .deepLearningUtils import *
from scipy.spatial import distance, distance_matrix

# Loads in files for one prediction
def getData(filename, atom, cutoff=4):
    coords, aas = parsePDB(filename, atom=atom)   
    maps = distance_matrix(coords, coords)
    sep = seqsep(maps.shape[0])
    maps = transfomer(maps, cutoff=cutoff)
    return np.concatenate([np.expand_dims(maps, axis=-1), sep], axis=-1)
    
# Sequence separtion features
def seqsep(psize, normalizer=100, axis=-1):
    ret = np.ones((psize, psize))
    for i in range(psize):
        for j in range(psize):
            ret[i,j] = abs(i-j)*1.0/100-1.0
    return np.expand_dims(ret, axis)

def transfomer(X, cutoff=6, scaling=3.0):
    X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
    return np.arcsinh(X_prime)/scaling
