
from .particle import Particle
from .abstract_pop import Abstract_Pop
import numpy as np
from .helper_funcs import get_domination_status, dominates


class Population(Abstract_Pop):
    
    def __init__(self, objective_function, num_objectives = 2, 
                 num_variables = 1, num_particles = 100):
        self.num_objectives = num_objectives
        self.num_variables = num_variables
        self.objective_function = lambda x: np.array(objective_function(x)).reshape(-1,)
        self.particles = [Particle(num_variables, num_objectives)
                          for i in range(num_particles)]
        
        self.global_best_position = np.zeros(num_variables)
        self.global_best_cost = np.zeros(num_objectives)
        self.num_particles_mutated = 0
    
    def update_costs(self):
        costs = np.array([self.objective_function(row) for row in self.position])
        self._setValues('cost', costs)
        self.update_personal_bests()

    def update_personal_bests(self):
        [instance.update_personal_best() for instance in self.particles]
    
    def initialize_population(self, varMins, varMaxs):
        self.variable_mins = np.array(varMins).reshape(-1,)
        self.variable_maxs = np.array(varMaxs).reshape(-1,)
        posis = self.variable_mins + np.multiply(np.random.rand(*self.position.shape),
                                                    (self.variable_maxs - self.variable_mins))
        self._setValues('position', posis)
        self.update_costs()
        self._setValues('best_position', self.position)
        self._setValues('best_cost', self.cost)
    
    def particle_repair(self):
        indices = self.position < self.variable_mins
        indices = indices.astype(int)
        self._setValues('position', self.position*(1 - indices) + indices*self.variable_mins)
        self._setValues('velocity', self.velocity*(1 - indices) - self.velocity*indices)
        
        indices = self.position > self.variable_maxs
        indices = indices.astype(int)
        self._setValues('position', self.position*(1 - indices) + indices*self.variable_maxs)
        self._setValues('velocity', self.velocity*(1 - indices) - self.velocity*indices)
    
    def update_domination(self):
        self._setValues('is_dominated', get_domination_status(self.cost))
        
    def mutate(self, mutation_value):
        count = 0
        for i in range(self.num_particles):
            if np.random.rand() < mutation_value:
                new_position = self.position[i, :]
                nVar = self.num_variables
                j = np.random.randint(0, nVar)
                dx = mutation_value*(self.variable_maxs - self.variable_mins)
                lb = new_position[j] - dx[j]
                ub = new_position[j] + dx[j]
                new_position[j] = lb + (ub - lb)*np.random.rand()
                new_position = np.minimum(new_position, self.variable_maxs)
                new_position = np.maximum(new_position, self.variable_mins)

                new_costs = self.objective_function(new_position)
                if dominates(self.cost[i], new_costs):
                    self.position[i] = new_position
                    self.cost[i] = new_costs
                    count += 1
        self.num_particles_mutated = count

    def eval_next_iteration(self, optimObject, repObject):

        glob_learn_coeff = optimObject.global_learn_coeff
        persn_learn_coeff = optimObject.personal_learn_coeff
        inertia_weight = optimObject.inertia_weight
        leader_positions = repObject.select_leader(self)
        velos = (inertia_weight*self.velocity +
                         glob_learn_coeff*np.multiply(np.random.rand(1, self.num_variables), leader_positions - self.position) +
                         persn_learn_coeff*np.multiply(*np.random.rand(1, self.num_variables), self.best_position - self.position))
        
        self._setValues('velocity', velos)
        pos = self.position + velos
        self._setValues('position', pos)
        self.mutate(optimObject.mutation_value)
        self.particle_repair()
        self.update_costs()