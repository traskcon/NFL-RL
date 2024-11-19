'''Store common classes
TODO: Remove scaffolding and rewrite to be GRIDIRON specific'''
import numpy as np

class Agent():
    def __init__(self):
       #TODO: Add other common parameters
       self.defense = False
       self.position = None
       self.ballcarrier = False
       self.oob = False
       self.location = np.array([None, None])
       self.collide = True
       self.color = None
       self.name = ""


class World():
    def __init__(self):
        self.agents = []
        self.landmarks = []
        #Future parameters added as needed to support continuous movement