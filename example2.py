from mopso import *

func = lambda x: (1/x[0], x[0]**2 - x[1])


pop = Population(func, num_objectives = 2, num_variables = 2, num_particles = 20)

pop.initialize_population([0.1, 0.5], [1, 2])

rep = Repository(repository_size = 50)
PSO = Optim(personal_learn_coeff = 1, global_learn_coeff = 2, max_iterations = 50, mutation_rate=0.1)

PSO.solve(pop, rep)