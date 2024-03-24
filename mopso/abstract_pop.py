# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:45:46 2024

@author: Radhe
"""

from abc import ABC, abstractmethod
import numpy as np

class Abstract_Pop(ABC):
    
    def _getValues(self, name):
        return np.array([getattr(instance, name) for instance in self.particles])

    def _setValues(self, name, value):
        for i, val in zip(range(self.num_particles), value.reshape(self.num_particles,-1)):
            setattr(self.particles[i], name, val)
    
    @property
    def num_particles(self):
        return len(self.particles)
    
    @property
    def position(self):
        return self._getValues('position')

    @property
    def best_position(self):
        return self._getValues('best_position')

    @property
    def velocity(self):
        return self._getValues('velocity')
    
    @property
    def cost(self):
        return self._getValues('cost')
    
    @property
    def is_dominated(self):
        return self._getValues('is_dominated')
            
    def __getitem__(self, key):
        return self.particles[key]
    
    @abstractmethod
    def eval_next_iteration():
        pass