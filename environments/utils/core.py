'''Store common classes'''
import numpy as np

class Entity():
    def __init__(self):
        self.name = ""
        self.movable = False
        self.collide = True
        self.color = None
        self.location = None

class Agent(Entity):
    def __init__(self):
       super().__init__()
       self.movable = True 
       self.defense = False
       self.position = None
       self.ballcarrier = False

class World():
    def __init__(self):
        self.agents = []
        self.landmarks = []
        #Future parameters added as needed to support continuous movement

class Landmark(Entity):
    def __init__(self):
        super().__init__()