

import numpy as np
from .helper_funcs import dominates



class Particle():
    
    def __init__(self, num_variables, num_objectives):
        self.position = np.zeros(num_variables)
        self.velocity = np.zeros(num_variables)
        self.cost = np.zeros(num_objectives)
        
        self.best_position = np.zeros(num_variables)
        self.best_cost = np.zeros(num_objectives)
        self.is_dominated = False
        
    def update_personal_best(self):
        if dominates(self.cost, self.best_cost):
            self.best_position = self.position
            self.best_cost = self.cost
    
    def __lt__(self, other):
        if isinstance(other, Particle):
            return dominates(self.cost, other.cost)
    
    def __repr__(self):
        return f'Particle: Pos = {self.position} ;Cst = {self.cost}; Bests - Pos = {self.best_position} ; Cst: {self.best_cost} '
    

