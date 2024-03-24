
import matplotlib.pyplot as plt


class Optim():
    
    def __init__(self, personal_learn_coeff = 1, global_learn_coeff = 2,
                 inertia_weight = 0.9, inertia_damping = 0.4,
                 max_iterations = 100, mutation_rate = 0.2):
        
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.inertia_weight = inertia_weight
        self.inertia_damping = inertia_damping
        self.personal_learn_coeff = personal_learn_coeff
        self.global_learn_coeff = global_learn_coeff
        self.mutation_rate = mutation_rate
        self.mutation_func = lambda x: pow((1 - x/(max_iterations - 1)), 1/mutation_rate)
        self.mutation_value = 1
        self.interia_func = lambda x: ((inertia_weight - inertia_damping)*(1 - x)
                                           /(max_iterations - 1) + inertia_weight)
        
    def eval_next_iteration(self, population, repository):
        population.eval_next_iteration(self, repository)
        repository.eval_next_iteration(population)                   
        return population, repository
    
    def solve(self, population, repository):
        
        repository.eval_next_iteration(population)
        for i in range(self.max_iterations):
            population, repository = self.eval_next_iteration(population, repository)
            self.current_iteration += 1
            self.inertia_weight = self.interia_func(self.current_iteration)
            self.mutation_value = self.mutation_func((self.current_iteration))
            print('CurrIter: ', self.current_iteration, 'RepSize', repository.num_particles,
                  'Inertia: ', self.inertia_weight, 'Num_part.Muta', population.num_particles_mutated)
        self.setup_figure(population, repository)
        
    def setup_figure(self, pop, rep):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(pop.cost[:,0], pop.cost[:,1], 'X')
        ax1.set_title('Population')
        ax1.set_xlabel('Cost 1')
        ax1.set_ylabel('Cost 2')
        
        ax2.plot(rep.cost[:,0], rep.cost[:,1], 'o')
        ax2.set_title('Repository')
        ax2.set_xlabel('Cost 1')
        ax2.set_ylabel('Cost 2')
        plt.show()
