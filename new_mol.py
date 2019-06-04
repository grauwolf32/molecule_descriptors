import numpy as np
import os
import re

from mesh import *

class Molecule(object):
    def __init__(self, mesh, charge_points, charge_vals, wdv_points, wdv_vals):
        self.mesh = mesh
        self.wdv_points = wdv_points
        self.wdv_vals = wdv_vals
        self.charge_points = charge_points
        self.charge_vals = charge_vals
        self.graph_properties = list()

def gen_mol(filename):
    prefix = ".".join(filename.split(".")[:-1])
    
    if not os.path.exists(prefix + ".wrl"):
        print (prefix + ".wrl")
        print ("Could not find surface description for {}".format(filename))
        return
    
    mesh = read_mesh_surface(prefix + ".wrl")
    
    if not os.path.exists(prefix + ".ch"):
        print ("Could not find charges for {}".format(filename))
        return
    
    charge_points = list()
    charge_vals   = list()
    
    with open(prefix + ".ch") as f:
        for line in f:
            tmp = line.strip(', \n').split(' ')
            charge_points.append(np.array([float(p) for p in tmp[:-1]]))
            charge_vals.append(float(tmp[-1]))
            
    if not os.path.exists(prefix + ".wdv"):
        print ("Could not find wan der vaals radious description for {}".format(filename))
        return
    
    wdv_points = list()
    wdv_vals   = list()
    
    with open(prefix + ".wdv") as f:
        for line in f:
            tmp = line.strip(', \n').split(' ')
            wdv_points.append(np.array([float(p) for p in tmp[:-1]]))
            wdv_vals.append(float(tmp[-1]))
            
    mol  = Molecule(mesh, charge_points, charge_vals, wdv_points, wdv_vals)
    
    return mol

def CalculatePotential(mol, vert_ind, probe_radius):
    normal = mol.mesh.normals[vert_ind] / np.linalg.norm(mol.mesh.normals[vert_ind])
    point = mol.mesh.vertices[vert_ind] + probe_radius * normal
        
    n = len(mol.charge_points)
    force = np.zeros(3)
            
    for i in range(0, n):
        curr_vec = point -  mol.charge_points[i]
        inv_dist = 1.0 / (np.linalg.norm(curr_vec) + 1e-5)
        curr_dir = curr_vec * inv_dist
        
        force += curr_dir * mol.charge_vals[i] * (inv_dist * inv_dist)
            
    return np.inner(force, normal)
        
def CalculateLennardJonesPotential(mol, vert_ind, probe_radius):
    normal = mol.mesh.normals[vert_ind] / np.linalg.norm(mol.mesh.normals[vert_ind])
    point = mol.mesh.vertices[vert_ind] + probe_radius * normal
    n = len(mol.wdv_points)
        
    potential = 0.0
            
    for i in range(0, n):
        curr_vec =  point - mol.wdv_points[i]
        inv_dist = 1.0 / (np.linalg.norm(curr_vec) + 1e-5)
            
        sigma = (mol.wdv_vals[i]  + probe_radius) / 2.0
        incl = (sigma * inv_dist)**12 - (sigma * inv_dist)**6
        potential += incl
        
    return potential

def getProperties(mol, probe_radius):
    prop = dict({"electric_potential" : [], 
                "lennard_jones_potential" : [], 
                "mean_curvature": [],
                "gaussian_curvature": []})

    for vert_ind in range(0, len(mol.mesh.vertices)):
        # Calculate properties in point of the molecular surface
            
        prop["electric_potential"].append(CalculatePotential(mol, vert_ind, probe_radius))
        prop["lennard_jones_potential"].append(CalculateLennardJonesPotential(mol, vert_ind, probe_radius))
        prop["mean_curvature"].append(getMeanCurvature(mol.mesh, vert_ind))
        prop["gaussian_curvature"].append(getGaussianCurvature(mol.mesh, vert_ind))

    return prop

def prepare_data(prop):
    mol_data = np.asarray(list(prop.values()))
    mol_data = np.transpose(mol_data)
    shape = mol_data.shape
    
    mean = np.mean(mol_data, axis=0)
    var  = np.var(mol_data, axis=0)
    mol_data = (mol_data - mean) / np.sqrt(var)
    return mol_data, mean, var


def calc_projection(mesh, data):
    shape = data.shape
    
    M = np.zeros((shape[1], shape[1]))
    for i in range(0, shape[0]):
        neigh_triangles = mesh.FindTriangleNeighborsOfVertex(i)
        total_area = 0.0
        for t in neigh_triangles:
            total_area += mesh.getArea(t)
        
        M += np.outer(data[i], data[i])*total_area
        
    M /= shape[0]
    return M