'''Store common classes'''
import numpy as np

class Agent():
    def __init__(self):
       self.defense = False
       self.position = None
       self.ballcarrier = False
       self.oob = False
       self.location = np.array([None, None])
       self.collide = True
       self.color = None
       self.name = ""
       self.strength = 0


class World():
    def __init__(self):
        self.agents = []
        self.landmarks = []
        self.yardline = None
        #Future parameters added as needed to support continuous movement


class Team():
    def __init__(self, name):
        self.name = name
        self.score = 0
        self.playbook = dict()
        self.defense = False