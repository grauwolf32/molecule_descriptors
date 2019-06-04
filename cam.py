from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class CameraBase(object):
    """camera.Base camera object all other inherit from..."""
    def __init__(self, pos=[0,0,0], rotation=[0,0,0]):
        """create the camera
           pos = position of the camera
           rotation = rotation of camera"""
        self.posx, self.posy, self.posz = pos
        self.rotx, self.roty, self.rotz = rotation

    def push(self):
        """Activate the camera - anything rendered after this uses the cameras transformations."""
        glPushMatrix()

    def pop(self):
        """Deactivate the camera - must be called after push or will raise an OpenGL error"""
        glPopMatrix()

    def get_pos(self):
        """Return the position of the camera as a tuple"""
        return self.posx, self.posy, self.posz

    def set_pos(self, pos):
        """Set the position of the camera from a tuple"""
        self.posx, self.posy, self.posz = pos

    def get_rotation(self):
        """Return the rotation of the camera as a tuple"""
        return self.rotx, self.roty, self.rotz

    def set_facing_matrix(self):
        """Transforms the matrix so that all objects are facing camera - used in Image3D (billboard sprites)"""
        pass

    def set_skybox_data(self):
        """Transforms the view only for a skybox, ie only rotation is taken into account, not position"""
        pass

class LookFromCamera(CameraBase):
    """camera.LookFromCamera is a FPS camera"""
    def __init__(self, pos=(0,0,0), rotation=(0,0,0)):
        CameraBase.__init__(self, pos, rotation)

    def push(self):
        glPushMatrix()
        glRotatef(self.rotx, 1, 0, 0)
        glRotatef(self.roty, 0, 1, 0)
        glRotatef(self.rotz, 0, 0, 1)
        glTranslatef(-self.posx, -self.posy, self.posz)

    def pop(self):
        glPopMatrix()

    def get_pos(self):
        return self.posx, self.posy, self.posz

    def get_rotation(self):
        return self.rotx, self.roty, self.rotz

    def set_facing_matrix(self):
        glRotatef(-self.rotz, 0, 0, 1)
        glRotatef(-self.roty, 0, 1, 0)
        glRotatef(-self.rotx, 1, 0, 0)

    def set_skybox_data(self):
        glRotatef(self.rotx, 1, 0, 0)
        glRotatef(self.roty, 0, 1, 0)
        glRotatef(self.rotz, 0, 0, 1)

class LookAtCamera(CameraBase):
    """camera.LookAtCamera is a third-person camera"""
    def __init__(self, pos=[0,0,0], rotation=[0,0,0],
                 distance=0):
        """create the camera
           pos is the position the camera is looking at
           rotation is how much we are rotated around the object
           distance is how far back from the object we are"""
        CameraBase.__init__(self, pos, rotation)
        self.distance = distance

    def push(self):
        glPushMatrix()
        glTranslatef(0, 0, -self.distance)
        glRotatef(-self.rotx, 1, 0, 0)
        glRotatef(-self.roty, 0, 1, 0)
        glRotatef(self.rotz, 0, 0, 1)
        glTranslatef(-self.posx, -self.posy, self.posz)

    def set_facing_matrix(self):
        glRotatef(-self.rotz, 0, 0, 1)
        glRotatef(self.roty, 0, 1, 0)
        glRotatef(self.rotx, 1, 0, 0)

    def set_skybox_data(self):
        glRotatef(-self.rotx, 1, 0, 0)
        glRotatef(-self.roty, 0, 1, 0)
        glRotatef(self.rotz, 0, 0, 1)
