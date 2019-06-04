#!/usr/bin/env python3

import os
import re
import json
import argparse
import numpy as np


from new_mol import *
from mesh import *

def getLocExtremums(mesh, function, dist):
    mins = []
    maxs = []

    for v in range(0, len(mesh.vertices)):
        neighbors = list(filter(lambda u: mesh.dist[u][v] <= dist, mesh.G.nodes()))
        neighbors.remove(v)
        
        n_vals = np.asarray([function[u] for u in neighbors])
        n_min = np.min(n_vals)
        n_max = np.max(n_vals)

        if function[v] < n_min:
            mins.append(v)
        elif function[v] > n_max:
            maxs.append(v)

    return mins, maxs

def findFragments(mesh, function, dist, sigmas):
    function = np.asarray(function)
    if len(function.shape) > 1:
        prop_len = function.shape[1]
    else:
        prop_len = 1

    blurred_func = FilterMesh(mesh, 
                              GaussianKernelWeightedDist, 
                              function, 
                              sigma=dist, 
                              prop_len=prop_len)
    
    blurred_func = np.asarray(blurred_func)
    fragments = dict()

    for sigma in sigmas:
        print("sigma: {}".format(sigma))
        log_func = FilterMesh(mesh,
                              LoGKernelWeightedDist,
                              blurred_func,
                              sigma=sigma,
                              prop_len=prop_len)
        
        mins, maxs = getLocExtremums(mesh, log_func, sigma)
        fragments[sigma] = [mins, maxs]
    
    return fragments

def main():
    parser = argparse.ArgumentParser(description='Sp')
    parser.add_argument('-f', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.f):
        print("File {} not found".format(args.f))
        return

    filename = args.f

    m = gen_mol(filename)
    base_fn = "".join(filename.split(".")[:-1])
    enc_prop = []
    with open("".join((base_fn, ".scprop")), "r") as f:
        for line in f:
            line = line.strip(" \n")
            if not line:
                continue
            enc_prop.append(float(line))
    
    enc_prop = np.asarray(enc_prop)

    prj_prop = []
    with open("".join((base_fn, ".proj")), "r") as f:
        for line in f:
            line = line.strip(" \n")
            if not line:
                continue
            prj_prop.append(float(line))
    
    prj_prop = np.asarray(prj_prop)
    prj_prop = [(d - np.mean(prj_prop))/np.sqrt(np.var(prj_prop)) for d in prj_prop]

    frag_enc = findFragments(m.mesh, enc_prop, dist=0.25, sigmas=[0.5, 0.6, 0.7, 0.8])
    frag_prj = findFragments(m.mesh, prj_prop, dist=0.25, sigmas=[0.5, 0.6, 0.7, 0.8])

    with open("".join((base_fn, ".spproj")), "w") as f:
        f.write(json.dumps(frag_prj))

    with open("".join((base_fn, ".frag_enc")), "w") as f:
        f.write(json.dumps(frag_enc))

    return

if __name__ == "__main__":
    main()