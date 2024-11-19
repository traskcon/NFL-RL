'''Store common classes
TODO: Remove scaffolding and rewrite to be GRIDIRON specific'''
import numpy as np

class Entity():
    def __init__(self):
        self.name = ""
        self.collide = True
        self.color = None
        self.location = None

class Agent(Entity):
    def __init__(self):
       super().__init__()
       #TODO: Add other common parameters
       self.defense = False
       self.position = None
       self.ballcarrier = False
       self.oob = False


class World():
    def __init__(self):
        self.agents = []
        self.landmarks = []
        #Future parameters added as needed to support continuous movement