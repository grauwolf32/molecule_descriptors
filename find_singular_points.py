#!/usr/bin/python2.7

import os
import re
import argparse
import numpy as np
import networkx as nx
import tensorflow as tf

from enum import Enum
from copy import deepcopy

from collections import deque
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr, lsmr

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def getTransitionMatrixToBasisWithNormalVector(normal):
    normal = normal / np.linalg.norm(normal)
    b1 = (np.eye(3) - np.outer(normal, normal))[:,0]
    b1 = b1 / (np.linalg.norm(b1) + 1e-7)
    
    b2 = normal
    b3 = np.cross(b1, normal)
    b3 = b3 / (np.linalg.norm(b3))
    
    m =  np.transpose(np.array([b1,b2,b3]))
    return m

def getShortestRotationMatrixSpecial(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-7)
    v2 = v2 / (np.linalg.norm(v2) + 1e-7)
    
    v_cross = np.cross(v1, v2)
    v_cross = v_cross / np.linalg.norm(v_cross)
    
    v3 = np.cross(v1, v_cross)
    v3 = v3 / (np.linalg.norm(v3) + 1e-7)
    
    sin_alpha = -np.inner(v2, v3)
    cos_alpha =  np.inner(v1, v2)
    
    rot_mat = np.array([[0.0]*3]*3)
    
    rot_mat[0][0] =  cos_alpha
    rot_mat[0][2] = -sin_alpha
    rot_mat[2][0] =  sin_alpha
    rot_mat[2][2] =  cos_alpha
    rot_mat[1][1] =  1.0
    
    m = np.array([[0.0]*3] * 3)
    m[:, 0] = v3
    m[:, 1] = v_cross
    m[:, 2] = v1
    
    rotation_matrix = np.dot(m, np.dot(rot_mat, np.linalg.inv(m)))
    return rotation_matrix

    tangent_basis = mesh.tangent_basises[vert_ind]
    neibors = list(mesh.GetVerticesWithinDist(vert_ind, dist)) # Neighbors + adjacent ?
    neibors.append(vert_ind) # last neighbor is current point
        
    n_neibors = len(neibors)
    
    coeff = np.array([np.array([0.0]*2)]*n_neibors)
    deriv = np.array([0.0]*n_neibors)
        
    coeff[-1][0] = 0.0
    coeff[-1][1] = 0.0
    deriv[-1] = 0.0
        
    curr_vertice = mesh.vertices[vert_ind]
        
    for i in xrange(0, n_neibors-1):
        n_vertice = mesh.vertices[neibors[i]]
        projected_vect = np.dot(tangent_basis, n_vertice - curr_vertice)
           
        coeff[i][0] = projected_vect[0]
        coeff[i][1] = projected_vect[1]
            
        pr_length = np.linalg.norm(projected_vect) + 1e-7
        deriv[i] = (funct[neibors[i]] - funct[vert_ind]) / pr_length
            
    grad = np.linalg.lstsq(coeff, deriv)[0]#np.dot(np.pinv(coeff), deriv)
        
    if gradient_constrains:
            grad[0] = min(max(gradient_constrains[0], grad[0]),gradient_constrains[1])
            grad[1] = min(max(gradient_constrains[0], grad[1]),gradient_constrains[1])
            
    return grad

def CalculateHessianEigenvalsRatio(mesh, function, dist, gradient_treshold=1e-2 ,gradient_constrains=None):
    gradients = [CalculateFunctionGradient(mesh, i, function, dist) for i in xrange(0, mesh.num_points)]
    
    f_dx = [gradients[i][0] for i in xrange(0, mesh.num_points)]
    f_dy = [gradients[i][1] for i in xrange(0, mesh.num_points)]
    
    grad_norm = np.sqrt([f_dx[i]**2 + f_dy[i]**2 for i in xrange(0, mesh.num_points)])
    
    gradients_fdx = [CalculateFunctionGradient(mesh, i, f_dx, dist) for i in xrange(0, mesh.num_points)]
    gradients_fdy = [CalculateFunctionGradient(mesh, i, f_dy, dist) for i in xrange(0, mesh.num_points)]
    
    f_dxdx = [gradients_fdx[i][0] for i in xrange(0, mesh.num_points)]
    f_dxdy = [gradients_fdx[i][1] for i in xrange(0, mesh.num_points)]
    
    f_dydx = [gradients_fdy[i][0] for i in xrange(0, mesh.num_points)]
    f_dydy = [gradients_fdy[i][1] for i in xrange(0, mesh.num_points)]
    
    eigenvals = []
    
    for i in xrange(0, mesh.num_points):
        H = np.array([[0.0]*2]*2)
        
        H[0][0] = f_dxdx[i]
        H[0][1] = f_dxdy[i]
        H[1][0] = f_dydx[i]
        H[1][1] = f_dydy[i]
        
        eigenvals.append(np.linalg.eigvals(H))
        
    ratios = []
    for i in xrange(0, mesh.num_points):
        e = eigenvals[i]
        max_e = np.max([np.abs(e[0]), np.abs(e[1])])
        min_e = np.min([np.abs(e[0]), np.abs(e[1])])
        
        if min_e < 1e-7:
            ratios.append(max_e / 1e-7)
            continue
        
        if grad_norm[i] < gradient_treshold:
            ratios.append(0.0)
            continue
            
        ratios.append(max_e / min_e)
        
    return ratios, eigenvals

def GetScalarPropertiesAsMainPCAComponent(mesh, properties):
    properties_scaler = StandardScaler()
    scaled_properties = properties_scaler.fit_transform(properties)
    
    n_components = properties.shape[1]
    properties_pca = PCA(n_components = n_components)
    properties_pca.fit(scaled_properties)
    
    main_component = properties_pca.components_[0]
    exp_var_ratio  = properties_pca.explained_variance_ratio_[0]
    
    print "main component {}".format(main_component)
    print "explained variance ratio {}".format(exp_var_ratio)
    
    projected_properties = np.array([0.0]*mesh.num_points)
    for i in xrange(0, mesh.num_points):
        projected_properties[i] = np.inner(main_component, properties[i])
        
    return projected_properties

def CalculateLoG(mesh, function, sigma):
    filtered_function = FilterMesh(mesh,LoGKernelWeightedDist, function, sigma=sigma, prop_len = 1)
    return filtered_function

def CalculateLocalMaximumsAndMinimumsOnLevels(mesh, function, dist):
    maximums = dict()
    minimums = dict()
    
    n_levels = function.shape[0]
    
    for vert_ind in xrange(0, mesh.num_points):
        for current_level in xrange(0, n_levels):
            current_val = function[current_level][vert_ind]
            
            is_max = True
            is_min = True
        
            neighbors = mesh.GetVerticesWithinDist(vert_ind, dist[current_level])
            neighbors.discard(vert_ind)
        
            for neighbor in neighbors:
                neighbor_val = function[current_level][neighbor]
            
                if neighbor_val <= current_val:
                    is_min = False
            
                if neighbor_val >= current_val:
                    is_max = False
                
                if  not (is_max or is_min):
                    break
                
            if is_max:
                maximums.setdefault(vert_ind, []).append((current_level, current_val))
            
            if is_min:
                minimums.setdefault(vert_ind, []).append((current_level, current_val))
            
    return maximums, minimums

class LoGKernelWeightedDist(object):
    def __init__(self, **kwargs):
        self.sigma = kwargs.setdefault("sigma", 0.5)
        self.prop_len = kwargs.setdefault("prop_len", 1)
        
        self.coeff_sum = 0.0
        self.prop_sum  = np.array([0.0] * self.prop_len)
            
        self.norm_coeff = 1.0 / (np.pi * self.sigma**4)
        self.sigma_square = self.sigma**2
                                 
    def add(self, dist, weight, properties):
        arg = dist**2 / (2.0 * self.sigma_square)
        coeff = weight * (1.0 - arg) * np.exp(-arg) * (self.norm_coeff)
        
        self.coeff_sum += np.abs(coeff)
        self.prop_sum   = self.prop_sum + coeff * properties
        
    def getResult(self):
        return self.prop_sum * (1.0 / self.coeff_sum)
    
    def getRadius(self):
        return self.sigma*3.0
    
class GaussianKernelWeightedDist(object):
    def __init__(self, **kwargs):
        self.sigma = kwargs.setdefault("sigma", 0.5)
        self.prop_len = kwargs.setdefault("prop_len", 1)
        
        self.coeff_sum = 0.0
        self.prop_sum  = np.array([0.0] * self.prop_len) 
        self.sigma_square = self.sigma**2
                                 
    def add(self, dist, weight, properties):
        arg = dist**2 / (2.0 * self.sigma_square)
        coeff = weight * np.exp( -dist**2 / (2.0 * self.sigma_square))
        
        self.coeff_sum += coeff
        self.prop_sum   = self.prop_sum + coeff * properties
        
    def getResult(self):
        return self.prop_sum * (1.0 / self.coeff_sum)
    
    def getRadius(self):
        return self.sigma*3.0
    
def FilterMesh(mesh, kernel, properties, **kwargs):
    properties_filtered = np.copy(properties)
    
    for vert_ind in xrange(0, mesh.num_points):
        current_kernel = kernel(**kwargs)
        radius = current_kernel.getRadius()
        
        vertices_within_distance = mesh.GetVerticesWithinDist(vert_ind, radius)
        
        for i in vertices_within_distance:
            triangles_area = [mesh.triangles_area[tr] for tr in mesh.FindTriangleNeighboursOfVertex(i)]
            weight = np.sum(triangles_area) / len(triangles_area)
            
            current_kernel.add(mesh.geodesic_dist[i][vert_ind], weight, properties[i])
            
        properties_filtered[vert_ind] = current_kernel.getResult()
        
    return properties_filtered


def getConatgens(p1, p2, p3):
    # alpha is the angle between p1p2 and p2p3
    
    v1 = p1 - p2 
    v2 = p3 - p2
    
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    sin_alpha =  np.linalg.norm(np.cross(v1,v2)) / (v1_norm * v2_norm + 1e-7)
    cos_alpha =  np.inner(v1,v2) / (v1_norm * v2_norm + 1e-7)
    
    return cos_alpha / (sin_alpha + 1e-7)
    
class Mesh(object):
    def __init__(self, vertices, normals, triangles):
        self.vertices = vertices # list of numpy arrays
        self.normals  = normals  # list of numpy arrays
        self.triangles = triangles 
        
        self.num_points = len(self.vertices)
        self.num_triangles = len(self.triangles)
        
        # Calculate list of triangles for every vertice
        self.vertice_triangles = dict()
        for i in xrange(0, self.num_points):
            self.vertice_triangles[i] = set()
            
        for t in xrange(0, self.num_triangles):
            for v in self.triangles[t]:
                self.vertice_triangles[v].add(t)
            
        # Calculate triangles area
        self.triangles_area = np.array([0.0]*len(self.triangles))
        for t in xrange(0, self.num_triangles):
            v1 = self.vertices[self.triangles[t][1]] - self.vertices[self.triangles[t][0]]
            v2 = self.vertices[self.triangles[t][2]] - self.vertices[self.triangles[t][0]]
            area = (1.0/2.0)*np.linalg.norm(np.cross(v1,v2))
            
            self.triangles_area[t] = area   
            
        # Build surface graph
        self.G = nx.Graph() 
        self.G.add_nodes_from(xrange(0,self.num_points-1))
        
        for t in self.triangles:
            self.G.add_edge(t[0], t[1], weight=np.linalg.norm(self.vertices[t[0]] - self.vertices[t[1]]))
            self.G.add_edge(t[0], t[2], weight=np.linalg.norm(self.vertices[t[0]] - self.vertices[t[2]]))
            self.G.add_edge(t[1], t[2], weight=np.linalg.norm(self.vertices[t[1]] - self.vertices[t[2]]))
            
        # Tangent basises calculations
        self.tangent_basises = list()
        self.CalcConsistentTangentBasises()
        
        # Calculate distances for every two points in graph 
        dist_dijkstra_dict = {key : value for (key, value) in nx.all_pairs_dijkstra_path_length(self.G)}

        self.geodesic_dist = []
        for i in xrange(0, self.num_points):
            self.geodesic_dist.append(np.array([0.0]*self.num_points))
            
            for j in xrange(0, self.num_points):
                self.geodesic_dist[i][j] = dist_dijkstra_dict[i][j]
                
        self.euclidian_dist = []
        for i in xrange(0, self.num_points):
            self.euclidian_dist.append(np.array([0.0]*self.num_points))
            
            for j in xrange(0, self.num_points):
                self.euclidian_dist[i][j] = np.linalg.norm(self.vertices[i] - self.vertices[j])
                
        
    def FindTriangleNeighboursOfVertex(self, vert_ind):
        neighb_triangles = self.vertice_triangles[vert_ind]
                
        return list(neighb_triangles)
    
    def GetNeighborsWithinDist(self, vert_ind, dist):
        neighbors = self.G.neighbors(vert_ind)
        neighbors_within_dist = set()
        
        for neighbor in neighbors:
            if self.geodesic_dist[neigbor][vert_ind] <= dist:
                neighbors_within_dist.add(neighbor)
                
        neighbors_within_dist.discard(vert_ind)
        return neighbors_within_dist
    
    def GetNeighborsWithinDistWithAdjacent(self, vert_ind, dist):
        neighbors_within_dist = self.GetNeighborsWithinDist(vert_ind, dist)
        neighbors_within_dist_with_adjacent = set().union(neighbors_within_dist)
        
        for neighbor in neighbors_within_dist: 
            n_neighbors = self.G.neighbors(neighbor)
            neighbors_within_dist_with_adjacent = neighbors_within_dist_with_adjacent.union(set(n_neighbors))
            
        neighbors_within_dist_with_adjacent.discard(vert_ind)
            
        return neighbors_within_dist_with_adjacent
    
    def GetVerticesWithinDist(self, vert_ind, dist):
        visited_vertices = set()
        vertices_deque = deque()
        vertices_within_dist = []
        
        vertices_deque.append(vert_ind)
        visited_vertices.add(vert_ind)
        
        while len(vertices_deque) > 0:
            curr_vert = vertices_deque.pop()
            curr_vert_neigbors = self.G.neighbors(curr_vert)
            vertices_within_dist.append(curr_vert)
            
            for neighbor_ind in curr_vert_neigbors:
                if neighbor_ind not in visited_vertices:
                    if self.geodesic_dist[vert_ind][neighbor_ind] < dist:
                        vertices_deque.append(neighbor_ind)
                        visited_vertices.add(neighbor_ind)
        
        return set(vertices_within_dist)
    
    def GetVerticesWithinDistWithAdjacent(self, vert_ind, dist):
        vertices_within_dist = self.GetVerticesWithinDist(vert_ind, dist)
        neighbors = set()
        
        for curr_vert in vertices_within_dist:
            neighbors = neighbors.union(set(self.G.neighbors(curr_vert)))
            
        return neighbors.union(vertices_within_dist)
            
    
    def CalculateGaussianCurvature(self, vert_ind):
        neighb_triangles = self.FindTriangleNeighboursOfVertex(vert_ind)
        
        total_area  = 0.0
        total_angle = 0.0
        
        for t in neighb_triangles:
            tmp = set(self.triangles[t])
            tmp.remove(vert_ind)
            tmp = list(tmp)
            
            v1 = self.vertices[tmp[0]] - self.vertices[vert_ind]
            v2 = self.vertices[tmp[1]] - self.vertices[vert_ind]
            
            cos_alpha = np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
            
            area  = self.triangles_area[t]
            angle = np.arccos(min(1.0,max(-1.0, cos_alpha)))
            
            total_area  += area
            total_angle += angle
            
        return (2.0*np.pi - total_angle) / total_area
    
            
    def CalcConsistentTangentBasises(self):
        n_equations = 0
        for i in xrange(0, self.num_points):
            n_equations += self.G.degree(i)
        
        n_equations = 3*(n_equations + 1)
        n_variables  = self.num_points * 3
    
        matrix_row   = []
        matrix_col   = []
        matrix_value = []
        b = []
        
        # First equation is init value
        current_equation = 0
        rot_m = getTransitionMatrixToBasisWithNormalVector(self.normals[0])
        
        for i in xrange(0, 3):
            matrix_row.append(i)
            matrix_col.append(i)
            matrix_value.append(1.0)
            
            b.append(rot_m[i][0])
        
        current_equation += 1
        
        for vert_ind in xrange(0, self.num_points):
            for neighbor in self.G.neighbors(vert_ind):
                neighbor_normal = self.normals[neighbor]
                rot_m = getShortestRotationMatrixSpecial(self.normals[vert_ind], neighbor_normal)
                
                for curr_row in xrange(0, 3):
                    for curr_coord in xrange(0,3):
                        matrix_row.append(3*current_equation + curr_row)
                        matrix_col.append(3*vert_ind + curr_coord)
                        matrix_value.append(rot_m[curr_row][curr_coord])
                        
                    matrix_row.append(3*current_equation + curr_row)
                    matrix_col.append(3*neighbor + curr_row)
                    matrix_value.append(-1.0)
                    b.append(0)
                    
                current_equation += 1
                
        A = coo_matrix((matrix_value, (matrix_row, matrix_col)), shape=[n_equations, n_variables])
        result = lsmr(A, b)
        
        variables = result[0]
    
        for vert_ind in xrange(0, self.num_points):
            k = variables[3*vert_ind:3*(vert_ind+1)]
            
            curr_normal = self.normals[vert_ind]
            curr_normal = curr_normal / np.linalg.norm(curr_normal)
                      
            v2 = k - curr_normal * np.dot(k, curr_normal)
            v2 = v2 / np.linalg.norm(v2)
            
            v1 = np.cross(v2, curr_normal)
            v1 = v1 / np.linalg.norm(v1)
            
            basis = np.transpose(np.array([v1,v2,curr_normal]))
            self.tangent_basises.append(basis)
            
        return
                
    def CalculateMeanCurvature(self, vert_ind):
        neighb_triangles = self.FindTriangleNeighboursOfVertex(vert_ind)
        n = len(neighb_triangles)
        mean_curv = np.zeros(3) 
        
        adjacent_triangles = dict()
        adjacent_vertices = set()
        
        for i in xrange(0, n):
            for v_i in self.triangles[neighb_triangles[i]]:
                if v_i == vert_ind:
                    continue
                if v_i in adjacent_vertices:
                    adjacent_triangles[v_i].append(i)
                else:
                    adjacent_vertices.add(v_i)
                    adjacent_triangles[v_i] = list()
                    adjacent_triangles[v_i].append(i)
                
        for p in adjacent_vertices:
            p_triangles = adjacent_triangles[p]
            
            t1 = p_triangles[0]
            t2 = p_triangles[1]
            
            t1_p = [v for v in self.triangles[neighb_triangles[t1]] if v != p and v != vert_ind][0]
            t2_p = [v for v in self.triangles[neighb_triangles[t2]] if v != p and v != vert_ind][0]
            
            cot_ap = getConatgens(self.vertices[p], self.vertices[t1_p], self.vertices[vert_ind])
            cot_bp = getConatgens(self.vertices[p], self.vertices[t2_p], self.vertices[vert_ind])
        
            mean_curv += (cot_ap + cot_bp)*(self.vertices[vert_ind] - self.vertices[p])
            
        # Calculate area
        total_area = 0.0
        for t in neighb_triangles:
            v1 = self.vertices[self.triangles[t][1]] - self.vertices[self.triangles[t][0]]
            v2 = self.vertices[self.triangles[t][2]] - self.vertices[self.triangles[t][0]]
            
            area  = (1.0/2.0) * np.linalg.norm(np.cross(v1, v2))
            total_area += area
            
        mean_curv_val = np.linalg.norm(mean_curv) / (4.0 * total_area)
        return mean_curv_val

class MolecularGraph(object):
    def __init__(self, mesh, charge_points, charge_vals, wdv_points, wdv_vals, probe_radius):
        self.mesh = mesh
        self.wdv_points = wdv_points
        self.wdv_vals = wdv_vals
        self.charge_points = charge_points
        self.charge_vals = charge_vals
        
        self.graph_properties = list()
        
        for vert_ind in xrange(0, self.mesh.num_points):
            # Calculate properties in point of the molecular surface
            
            prop = dict()
            prop["electric_potential"] = self.CalculatePotential(vert_ind, probe_radius)
            prop["lennard_jones_potential"] = self.CalculateLennardJonesPotential(vert_ind, probe_radius)
            prop["mean_curvature"] = self.mesh.CalculateMeanCurvature(vert_ind)
            prop["gaussian_curvature"] = self.mesh.CalculateGaussianCurvature(vert_ind)
            
            self.graph_properties.append(prop)
            
    def CalculatePotential(self, vert_ind, probe_radius):
        normal = self.mesh.normals[vert_ind] / np.linalg.norm(self.mesh.normals[vert_ind])
        point = self.mesh.vertices[vert_ind] + probe_radius * normal
        
        n = len(self.charge_points)
        force = np.zeros(3)
            
        for i in xrange(0, n):
            curr_vec = point -  self.charge_points[i]
            inv_dist = 1.0 / (np.linalg.norm(curr_vec) + 1e-7)
            curr_dir = curr_vec * inv_dist
        
            force += curr_dir * self.charge_vals[i] * (inv_dist * inv_dist)
            
        return np.inner(force, normal)
        
    def CalculateLennardJonesPotential(self,vert_ind, probe_radius):
        normal = self.mesh.normals[vert_ind] / np.linalg.norm(self.mesh.normals[vert_ind])
        point = self.mesh.vertices[vert_ind] + probe_radius * normal
        
        n = len(self.wdv_points)
        
        potential = 0.0
            
        for i in xrange(0, n):
            curr_vec =  point - self.wdv_points[i]
            inv_dist = 1.0 / (np.linalg.norm(curr_vec) + 1e-7)
            
            sigma = (self.wdv_vals[i]  + probe_radius) / 2.0
            incl = (sigma * inv_dist)**12 - (sigma * inv_dist)**6
            potential += incl
        
        return potential
    
    def CalculateSingularPoints(self, sigmas, space_scale_sigma):
        properties = np.array([np.array(self.graph_properties[i].values()) for i in xrange(0, self.mesh.num_points)])
        scaled_properties = FilterMesh(self.mesh, GaussianKernelWeightedDist, properties, sigma=space_scale_sigma, prop_len=properties.shape[1])
       
        projected_properties = GetScalarPropertiesAsMainPCAComponent(self.mesh, scaled_properties)
        detect_func_on_levels = []
        
        for sigma in sigmas:
            curr_log = CalculateLoG(self.mesh, projected_properties, sigma)
            curr_log = curr_log * sigma**2
            detect_func_on_levels.append(curr_log)
            
            print "sigma: {}".format(sigma)
        
        detect_func_on_levels = np.array(detect_func_on_levels)
        
        print "Finding extremums over {} time steps".format(detect_func_on_levels.shape[0])
        
        maximums, minimums = CalculateLocalMaximumsAndMinimumsOnLevels(self.mesh, detect_func_on_levels, sigmas)
        
        max_points = []
        min_points = []
        
        for i in maximums:
            vert_maximums = maximums[i]
            if len(vert_maximums) > 1:
                vert_maximums = sorted(vert_maximums, key=lambda x: x[1], reverse=True)
                max_points.append((sigmas[vert_maximums[0][0]], vert_maximums[0][1], i))
                
        for i in minimums:
            vert_minimums = minimums[i]
            if len(vert_minimums) > 1:
                vert_minimums = sorted(vert_minimums, key=lambda x: x[1])
                min_points.append((sigmas[vert_minimums[0][0]], vert_minimums[0][1], i))
        
        return max_points, min_points, scaled_properties
    
def surf_reader(filename):
    if not os.path.exists(filename):
        print "File {} not exist!".format(filename)
        return
    
    point_regexp  = re.compile(r"point \[([^\]]+)\]")
    normal_regexp = re.compile(r"vector \[([^\]]+)\]")
    index_regexp  = re.compile(r"coordIndex \[([^\]]+)\]")
    
    with open(filename, "r") as f:
        file_data = f.read().strip()
        
    points  = re.findall(point_regexp, file_data)[0].strip().split("\n")
    normals = re.findall(normal_regexp, file_data)[0].strip().split("\n")
    meshidx = re.findall(index_regexp, file_data)[0].strip().split("\n")
    
    points = map(lambda x: np.array([float(p.strip(',')) for p in x.strip(',\n ').split()]), points)
    normals = map(lambda x: np.array([float(p.strip(',')) for p in x.strip(',\n ').split()]), normals)
    meshidx = map(lambda x: [int(p.strip(',')) for p in x.strip(',\n ').split()][:3], meshidx)
    
    return points, normals, meshidx

def gen_mol(filename, probe_radius=1.53):
    prefix = ".".join(filename.split(".")[:-1])
    
    if not os.path.exists(prefix + ".wrl"):
        print prefix + ".wrl"
        print "Could not find surface description for {}".format(filename)
        return
    
    points, normals, meshidx = surf_reader(prefix + ".wrl")
    
    if not os.path.exists(prefix + ".ch"):
        print "Could not find charges for {}".format(filename)
        return
    
    charge_points = list()
    charge_vals   = list()
    
    with open(prefix + ".ch") as f:
        for line in f:
            tmp = line.strip(', \n').split(' ')
            charge_points.append(np.array([float(p) for p in tmp[:-1]]))
            charge_vals.append(float(tmp[-1]))
            
    if not os.path.exists(prefix + ".wdv"):
        print "Could not find wan der vaals radious description for {}".format(filename)
        return
    
    wdv_points = list()
    wdv_vals   = list()
    
    with open(prefix + ".wdv") as f:
        for line in f:
            tmp = line.strip(', \n').split(' ')
            wdv_points.append(np.array([float(p) for p in tmp[:-1]]))
            wdv_vals.append(float(tmp[-1]))
            
    mesh = Mesh(points, normals, meshidx)
    mol  = MolecularGraph(mesh, charge_points, charge_vals, wdv_points, wdv_vals, probe_radius=probe_radius)
    
    return mol

def main():
    parser = argparse.ArgumentParser(description='Singular points')
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--probe_radius', type=float, default=0.05)
    parser.add_argument('--space_sigma' , type=float, default=0.33)

    parser.add_argument('--sigma_min', type=float, default=0.36)
    parser.add_argument('--sigma_max', type=float, default=0.64)
    parser.add_argument('--n_sigmas' , type=int,   default=5)


    args = parser.parse_args()
    if not os.path.exists(args.filename):
        print "Could not find file {}".format(filename)

    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.n_sigmas)

    print "Reading molecule data..."
    mol = gen_mol(args.filename, probe_radius=args.probe_radius)
    maximums, minimums, scaled_properties = mol.CalculateSingularPoints(sigmas, args.space_sigma)
    print "Done."

    singular_points = []
    singular_points_indexes = []

    for i in maximums:
        vert_ind = i[2]
        tmp = list(scaled_properties[vert_ind])
        
        point_area = 0.0
        neigbor_triangles =  mol.mesh.FindTriangleNeighboursOfVertex(vert_ind)
        for t in neigbor_triangles:
            point_area += mol.mesh.triangles_area[t]
        
        point_area = point_area / len(neigbor_triangles)
        tmp.append(point_area)
        
        point_radius = i[0] * np.sqrt(2.0)
        tmp.append(point_radius)
        tmp.append(1.0) # maximum

        singular_points.append(tmp)
        singular_points_indexes.append(vert_ind)

    for i in minimums:
        vert_ind = i[2]
        tmp = list(scaled_properties[vert_ind])
        
        point_area = 0.0
        neigbor_triangles =  mol.mesh.FindTriangleNeighboursOfVertex(vert_ind)
        for t in neigbor_triangles:
            point_area += mol.mesh.triangles_area[t]
        
        point_area = point_area / len(neigbor_triangles)
        tmp.append(point_area)
        
        point_radius = i[0] * np.sqrt(2.0)
        tmp.append(point_radius)
        tmp.append(-1.0) # minimums
        
        singular_points.append(tmp)
        singular_points_indexes.append(vert_ind)
    
    n_singular_points = len(singular_points)
    print "number of points: {}".format(n_singular_points)
    singular_points_file = ".".join(args.filename.split('.')[:-1]) + ".snp"

    print "Writing data to file {}".format(singular_points_file)
    with open(singular_points_file,"w") as f:
        for sp in singular_points:
            f.write(", ".join([str(i) for i in sp]) + "\n" )


    singular_points_pairs_file = ".".join(args.filename.split('.')[:-1]) + ".spp"
    print "Writing data to file {}".format(singular_points_pairs_file)

    with open(singular_points_pairs_file,"w") as f:
        for i in xrange(0, n_singular_points):
            for j in xrange(i+1, n_singular_points):
                tmp = list(np.copy(singular_points[i]))
                tmp = tmp + singular_points[j]

                p1_index = singular_points_indexes[i]
                p2_index = singular_points_indexes[j]
                
                euclidian_dist = mol.mesh.euclidian_dist[i][j]
                geodesic_dist  = mol.mesh.geodesic_dist[i][j]

                tmp.append(euclidian_dist)
                tmp.append(geodesic_dist)

                f.write(", ".join([str(prop) for prop in tmp]) + "\n" )


if __name__ == "__main__":
    main()