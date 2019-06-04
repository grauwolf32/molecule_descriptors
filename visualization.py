import os
import re
import time
import json

from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from pygame.locals import *
from OpenGL.arrays import vbo

from new_mol import *


import pygame as pg
import numpy as np  

from cam import *
from mesh import *

from encoders import * 

def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    return shader

def gen_shaders():
    vertex = create_shader(GL_VERTEX_SHADER, """
            varying vec4 vertex_color;
            void main(){
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                vertex_color = gl_Color;
            }""")

    fragment = create_shader(GL_FRAGMENT_SHADER, """
               varying vec4 vertex_color;
                void main() {
                gl_FragColor = vertex_color;
            }""")

    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)

    return program 

def init_window():
    pg.init()
    width = 860
    height = 640

    display = (width, height)
    pg.display.set_mode(display, HWSURFACE|OPENGL|DOUBLEBUF)
    gluPerspective(45, display[0]/display[1], 0.1, 30.0)
    glTranslatef(0.0, 0.0, -5)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    # Enable lighting
    #glEnable(GL_LIGHTING)
    # Set light model
    #glLightModelfv(GL_LIGHT_MODEL_AMBIENT, 0.33)
    # Enable light number 0
    #glEnable(GL_LIGHT0)
    #glEnable(GL_LIGHT1)
    # Set position and intensity of light
    #glLightfv(GL_LIGHT0, GL_POSITION, (0,0,-1))
    #glLightfv(GL_LIGHT0, GL_DIFFUSE, 0.33)

    # Setup the material
    
   # glEnable(GL_COLOR_MATERIAL)
   # glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    program = gen_shaders()
    #glUseProgram(program)


init_window()
cam = LookAtCamera(rotation=[90,0,0], distance=1.0)

def on_keypress(keys):
    if keys[K_UP]:
        cam.distance -= 0.2
    if keys[K_DOWN]:
        cam.distance += 0.2

    if keys[K_LEFT]:
        cam.roty -= 3
    if keys[K_RIGHT]:
        cam.roty += 3

    if keys[K_w]:
        cam.rotx += 3
    if keys[K_s]:
        cam.rotx -= 3

    if keys[K_q]:
        cam.rotz += 3
    if keys[K_a]:
        cam.rotz -= 3

    if keys[K_r]:
        cam.posx += 0.2
    if keys[K_f]:
        cam.posx -= 0.2

def draw_mesh(mesh, vertice_colors, wireframe=False):
    glPushMatrix()
    glEnable(GL_CULL_FACE)      
    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) 
    glBegin(GL_TRIANGLES)
    for t in mesh.triangles:
        for k in t:
            v = mesh.vertices[k]
            c = vertice_colors[k]
            glColor4f(c[0], c[1], c[2], 1.0)
            glVertex3f(v[0], v[1], v[2])
    glEnd()
    glDisable(GL_CULL_FACE)
    glPopMatrix()
    return  

def color_meshsp(mesh, base_color, sigma, extr, curvatures=None):
    vertice_colors = np.zeros((len(mesh.vertices), 3))        
    sp_color = np.random.uniform(0,1,3)
    sigma = float(sigma)
    radius = 1.44*sigma
    
    for v in extr:
        sp_color = np.random.uniform(0,1,3)
        neighb = list(filter(lambda u: mesh.dist[u][v] <= radius, mesh.G.nodes()))
        for n in neighb:
            vertice_colors[n] += sp_color
            
    for i in range(0, len(mesh.vertices)):
        if vertice_colors[i][0] == 0.0 and vertice_colors[i][1] == 0.0 and vertice_colors[i][2] == 0.0:
            if curvatures:
                vertice_colors[i] = curvatures[i]
            else:
                vertice_colors[i] = base_color

    return vertice_colors

def main():
    filename = "dataset/14.wrl"
    mesh = read_mesh_surface(filename)
    curvatures = np.asarray([getGaussianCurvature(mesh, i) for i in range(0, len(mesh.vertices))])
    #curvatures = np.sqrt(curvatures)
    min_curv = np.min(curvatures)
    max_curv = np.max(curvatures)
    mean_curv = np.mean(curvatures)

    #blurred_curvs = FilterMesh(mesh, GaussianKernelWeightedDist, curvatures, sigma=0.15, prop_len=1)
    #max_bcurv = np.max(blurred_curvs)
    #min_bcurv = np.min(blurred_curvs)

    print("Mean: {} Max: {} Min: {}".format(np.mean(curvatures), max_curv, min_curv))
    #print("Mean: {} Max: {} Min: {}".format(np.mean(blurred_curvs), max_bcurv, min_bcurv))
    scaled_curvs = np.asarray([min(1.0, (c - min_curv)/(max_curv-min_curv)) for c in curvatures])

    
    m = gen_mol(filename)
    prop = getProperties(m, probe_radius=0.20)
    data, mean, var = prepare_data(prop)
    M = calc_projection(m.mesh, data)
    eigval, eigvec  = np.linalg.eig(M)
    print(eigval[0]/np.sum(eigval))

    #proj_data = [np.dot(eigvec[0], data_vec) for data_vec in data]
    #sc_proj = [(a - np.min(proj_data))/(np.max(proj_data) - np.min(proj_data)) for a in proj_data]
    colors = []
    
    #model, enc, dec = get_autoencoder(n_layers=5,
    #                                  layer_sizes=[4, 32, 208, 192, 1],
    #                                  enc_activations=['relu', 'relu', 'relu', 'relu', 'linear'],
    #                                  dec_activations=['linear', 'relu', 'relu', 'relu', 'linear'],
    #                                  dropout=None)
    
    #model.compile(optimizer='adam', loss='mse', metrics=[r_square])
    #hist = model.fit(data, data, epochs=60, batch_size=64, shuffle=False, verbose=True, callbacks=[keras.callbacks.History()])
    
    #enc_data = np.asarray([i[0] for i in enc.predict(data)])
    #sc_enc = [(a - np.min(enc_data))/(1.5*np.mean(enc_data) - np.min(enc_data)) for a in enc_data]
    #print(sc_enc)

    m = gen_mol(filename)
    base_fn = filename.split(".")[0]
    enc_prop = []
    
    with open("".join((base_fn, ".encprop")), "r") as f:
        for line in f:
            line = line.strip(" \n")
            if not line:
                continue
            enc_prop.append(float(line))


    def colorize_data(data, base_color=np.asarray([0.33, 0.66, 0.1])):
        mean = np.mean(data)
        sigm = np.sqrt(np.var(data))
        sc_data = [(d-mean)/sigm for d in data]
        trunc_data = []

        for s in sc_data:
            if s > -1.5 and s < 1.5:
                trunc_data.append(s)
            elif s > 1.5:
                trunc_data.append(1.0)
            else:
                trunc_data.append(-1.0)

        trunc_data = (np.asarray(trunc_data) + 1)/2.0
        colors = [t*base_color for t in trunc_data]
        return colors
        

    #for pr in proj_data:
    #    colors.append(np.asarray([0.33, 0.66, 0.1])*pr)

    #for pr in sc_enc:
    #    colors.append(np.asarray([0.33, 0.66, 0.1])*min(1.0,max(-1.0, pr)))
    #colors = colorize_data(proj_data)

    #for curv in scaled_curvs:
    #    if curv > (min_curv)/(max_curv-min_curv):
    #        colors.append([0.1, 0.0, curv])
    #    else:
    #        colors.append([0.1, curv, 0.0])

    with open("test_sp.json", "r") as f:
        frag = json.loads(f.read())

    sigma = list(frag.keys())[1]
    colors = color_meshsp(m.mesh, np.asarray([0.33, 0.66, 0.1]), sigma, frag[sigma][0], curvatures=colorize_data(enc_prop))

    while True:
        for event in pg.event.get():
            if event == pg.QUIT: 
                pg.quit()
                quit()

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pg.quit()
                    quit()
            
        keys = pg.key.get_pressed()
        on_keypress(keys)

        glClearColor(1.0, 1.0, 1.0, 1.0) 
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        cam.push()

        draw_mesh(mesh, colors, wireframe=True)     

        cam.pop()
        pg.display.flip()
        pg.time.wait(20)

if __name__ == "__main__":
    main()