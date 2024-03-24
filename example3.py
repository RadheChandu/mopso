
from mopso import *



func = lambda x: (x**2, (x - 2)**2)

pop = Population(func, num_objectives = 2, num_variables = 1, num_particles = 10)

pop.initialize_population(-100000, 100000)

rep = Repository(repository_size = 100)
PSO = Optim(personal_learn_coeff = 0.1, global_learn_coeff = 0.2, max_iterations = 100, mutation_rate=0.1)

PSO.solve(pop, rep)
