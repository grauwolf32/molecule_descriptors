import os
import re

import numpy as np
import networkx as nx

def getCotangens(p1, p2, p3):
    # alpha is the angle between p1p2 and p2p3

    cot = 0.0
    v1 = p1 - p2 
    v2 = p3 - p2
    
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    p = v1_norm * v2_norm

    if p > 1e-5:
        sin_alpha =  np.linalg.norm(np.cross(v1,v2)) / (v1_norm * v2_norm)
        cos_alpha =  np.inner(v1,v2) / (v1_norm * v2_norm)
        if sin_alpha > 1e-5:
            cot = cos_alpha / sin_alpha
        else:
            cot = 10e5

    return cot

class Mesh(object):
    def __init__(self, vertices, normals, triangles):
        self.vertices = np.asarray(vertices)
        self.normals  = np.asarray(normals)
        self.triangles = triangles

        self.G = nx.Graph() 
        self.G.add_nodes_from(range(0, len(vertices)))
        
        for t in self.triangles:
            w1 = np.linalg.norm(self.vertices[t[0]] - self.vertices[t[1]])
            w2 = np.linalg.norm(self.vertices[t[0]] - self.vertices[t[2]])
            w3 = np.linalg.norm(self.vertices[t[1]] - self.vertices[t[2]])

            self.G.add_edge(t[0], t[1], weight=w1)
            self.G.add_edge(t[0], t[2], weight=w2)
            self.G.add_edge(t[1], t[2], weight=w3)

        self.dist = {key : value for (key, value) in nx.all_pairs_dijkstra_path_length(self.G)}


    def getArea(self, tr_tuple):
        p1,p2,p3 = tr_tuple
        v1 = self.vertices[p2] - self.vertices[p1]
        v2 = self.vertices[p3] - self.vertices[p1]
        area = (1.0/2.0)*np.linalg.norm(np.cross(v1,v2))
        return area

    def FindTriangleNeighborsOfVertex(self, vert_ind):
        neighb_vert = list(self.G.neighbors(vert_ind))
        neighb_triangles = list()
        
        for i in range(0, len(neighb_vert)):
            for j in range(i+1, len(neighb_vert)):
                neighb_triangles.append((vert_ind, neighb_vert[i], neighb_vert[j]))
                
        return neighb_triangles

def getGaussianCurvature(mesh, vert_ind):
    neighb_triangles = mesh.FindTriangleNeighborsOfVertex(vert_ind)
    total_area  = 0.0
    total_angle = 0.0
    curvature = 0.0

    for t in range(0 ,len(neighb_triangles)):
        tmp = list(filter(lambda x: x != vert_ind, neighb_triangles[t]))
        v1 = mesh.vertices[tmp[0]] - mesh.vertices[vert_ind]
        v2 = mesh.vertices[tmp[1]] - mesh.vertices[vert_ind]
            
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        p = l1*l2
         
        cos_alpha = 1.0
        if p > 1e-5:
            cos_alpha = np.inner(v1, v2) / p
            
        area  = mesh.getArea(neighb_triangles[t])
        angle = np.arccos(min(1.0, max(-1.0, cos_alpha)))
            
        total_area  += area
        total_angle += angle

    if total_area > 1e-5:
        curvature =  (2.0 * np.pi - total_angle) / total_area
        
    return curvature

def getMeanCurvature(mesh, vert_ind):
    neighb_triangles = mesh.FindTriangleNeighborsOfVertex(vert_ind)
    n = len(neighb_triangles)
    mean_curv = np.zeros(3) 
        
    surr_sides = []
    adj_vertis = set()
    
    for t in neighb_triangles:
        tmp = list(filter(lambda x: x != vert_ind, t))
        surr_sides.append(tmp)
        adj_vertis.add(tmp[0])
        adj_vertis.add(tmp[1]) 

    for v in adj_vertis:
        s = list(filter(lambda x: x[0] == v or x[1] == v, surr_sides))
        t = []
        
        for side in s:
            if side[0] != v:
                t.append(side[0])
            else:
                t.append(side[1])
        
        cot_ap = getCotangens(mesh.vertices[v], mesh.vertices[t[0]], mesh.vertices[vert_ind])
        cot_bp = getCotangens(mesh.vertices[v], mesh.vertices[t[1]], mesh.vertices[vert_ind])
        vec = (mesh.vertices[vert_ind] - mesh.vertices[v])
        mean_curv += (cot_ap + cot_bp)*vec
            
        # Calculate area
        
    total_area = 0.0
    for t in range(0 ,len(neighb_triangles)):
        area  = mesh.getArea(neighb_triangles[t])
        total_area  += area

    if total_area > 10e-5:
        mean_curv_val = np.linalg.norm(mean_curv) / (4.0 * total_area)
    else:
        mean_curv_val = 10e5

    return mean_curv_val


def read_mesh_surface(filename):
    if not os.path.exists(filename):
        print ("File {} not exist!".format(filename))
        return
    
    point_regexp  = re.compile(r"point \[([^\]]+)\]")
    normal_regexp = re.compile(r"vector \[([^\]]+)\]")
    index_regexp  = re.compile(r"coordIndex \[([^\]]+)\]")
    
    with open(filename, "r") as f:
        file_data = f.read().strip()
        
    points  = re.findall(point_regexp, file_data)[0].strip().split("\n")
    normals = re.findall(normal_regexp, file_data)[0].strip().split("\n")
    meshidx = re.findall(index_regexp, file_data)[0].strip().split("\n")
    
    points  = list(map(lambda x: [float(p.strip(',')) for p in x.strip(',\n ').split()], points))
    normals = list(map(lambda x: [float(p.strip(',')) for p in x.strip(',\n ').split()], normals))
    meshidx = list(map(lambda x: [int(p.strip(',')) for p in x.strip(',\n ').split()][:3], meshidx))

    mesh = Mesh(points, normals, meshidx)
    
    return mesh

class LoGKernelWeightedDist(object):
    def __init__(self, **kwargs):
        self.sigma = kwargs.setdefault("sigma", 0.5)
        self.prop_len = kwargs.setdefault("prop_len", 1)
        
        self.coeff_sum = 0.0
        self.prop_sum  = np.zeros(self.prop_len)
                                 
    def add(self, dist, weight, properties):
        arg = (dist**2) / (2.0 * self.sigma**2)
        coeff = (weight * (1.0 - arg) * np.exp(-arg)) / (np.pi * self.sigma**4)
        
        self.coeff_sum += np.abs(coeff)
        self.prop_sum  += coeff * properties
        
    def getResult(self):
        if self.coeff_sum > 10e-5:
            return self.prop_sum / self.coeff_sum

        elif self.prop_sum > 10e-5:
            return 10e5

        else:
            return 0.0
    
    def getRadius(self):
        return self.sigma * 3.0
    
class GaussianKernelWeightedDist(object):
    def __init__(self, **kwargs):
        self.sigma = kwargs.setdefault("sigma", 0.5)
        self.prop_len = kwargs.setdefault("prop_len", 1)
        
        self.coeff_sum = 0.0
        self.prop_sum  = np.zeros(self.prop_len)
                                 
    def add(self, dist, weight, properties):
        arg = dist**2 / (2.0 * self.sigma**2)
        coeff = weight * np.exp( -dist**2 / (2.0 * self.sigma**2))
        
        self.coeff_sum += coeff
        self.prop_sum  += coeff * properties
        
    def getResult(self):
        if np.abs(self.coeff_sum) > 1e-5:
            return self.prop_sum * (1.0 / self.coeff_sum)

        elif np.abs(self.prop_sum) > 1e-5:
            return 10e5
        
        else:
            return 0.0
    
    def getRadius(self):
        return self.sigma*3.0

def FilterMesh(mesh, kernel, properties, **kwargs):
    properties_filtered = np.copy(properties)
    
    for vert_ind in range(0, len(mesh.vertices)):
        #print(vert_ind, len(mesh.vertices))
        current_kernel = kernel(**kwargs)
        radius = current_kernel.getRadius()
        
        vertices_within_distance = list(filter(lambda u: mesh.dist[u][vert_ind] <= radius, mesh.G.nodes()))
        #print("vvd: {}".format(len(vertices_within_distance)))
        for i in vertices_within_distance:
            triangles_area = [mesh.getArea(tr) for tr in mesh.FindTriangleNeighborsOfVertex(i)]
            weight = np.sum(triangles_area) / len(triangles_area)
            
            current_kernel.add(mesh.dist[i][vert_ind], weight, properties[i])
            
        properties_filtered[vert_ind] = current_kernel.getResult()
        
    return properties_filtered