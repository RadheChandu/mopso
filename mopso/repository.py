
from .abstract_pop import Abstract_Pop

import copy
import numpy as np
from .helper_funcs import get_domination_status, get_leader_using_MDPL



class Repository(Abstract_Pop):

    def __init__(self, inflation_rate = 0.1, repository_size = 30,
                 grids_per_dimension = 100):
        self.repository_size = repository_size
        self.inflation_rate = inflation_rate
        self.grids_per_dimension = grids_per_dimension
        self.particles = []
    
    def __add__(self, other):
        self.particles.extend(copy.deepcopy(other.particles))
        return self
    
    @property
    def bins(self):
        return self._getValues('bins')
    
    @property
    def grid_index(self):
        return self._getValues('grid_index')
    
    def create_grid(self):
        costs = self.cost
        maxs = np.max(costs, axis = 0)
        mins = np.min(costs, axis = 0)
        diff = maxs - mins
        maxs = maxs + self.inflation_rate*diff
        mins = mins - self.inflation_rate*diff
        multiplier = np.repeat(np.linspace(0, 1, self.grids_per_dimension).reshape(-1,1), 
                               len(diff), axis = 1)
        
        return mins + (maxs - mins)*multiplier

    def find_grid_index(self):
        costs = self.cost
        num_objectives = costs.shape[1]
        num_grids = self.grids_per_dimension
        sub_grid_index = []
        bins = self.create_grid()
        for i in range(num_objectives): 
            sub_grid_index.append(np.digitize(costs[:,i], bins[:,i]))
        
        sub_grid_index = np.squeeze(sub_grid_index)
        grid_index = np.zeros(costs.shape[0])
        for row in sub_grid_index: 
            grid_index = grid_index*num_grids + row
        
        self._setValues('sub_grid_index', sub_grid_index.T)
        self._setValues('grid_index', grid_index.reshape(-1,1))
    
    def pop_indices(self, indices):
        [list.pop(self.particles, i) for i in sorted(indices, reverse=True)]
        
    def update_grid(self):
        
        grid_index = self.grid_index
        x, z = np.unique(grid_index, return_counts = True)
        vals = x[ z > 1]
        indices = []
        for val in vals:
            ind = np.where(grid_index == val)[0]
            indices.extend(np.random.choice(ind, size = len(ind) - 1, replace = False))
        
        if indices:
            self.pop_indices(indices)
    
    def delete_extra_elements(self):
        extra_num_elements =  self.num_particles - self.repository_size
        for i in range(extra_num_elements):
            var = sorted(self.grid_index.reshape(-1,))
            grid_vals = min(zip(var, var[1:]), key = lambda x: abs(x[0] - x[1]))
            val_to_delete = np.random.choice(grid_vals, 1)[0]
            ind = np.where(self.grid_index.reshape(-1,) == val_to_delete)[0][0]
            list.pop(self.particles, ind)
    
    def update_domination(self):
        self._setValues('is_dominated', get_domination_status(self.cost))
    
    def select_leader(self, populationObject, func = get_leader_using_MDPL):
        return get_leader_using_MDPL(self, populationObject)
    
    def eval_next_iteration(self, pop_object):

        self = self + pop_object
        self.update_domination()
        ind = np.where(np.invert(self.is_dominated))[0]
        if ind.size > 0:
            self.pop_indices(ind)
        self.create_grid()
        self.find_grid_index()
        self.update_grid()
        if self.num_particles > self.repository_size:
            self.delete_extra_elements()