from mopso import *





def func(x):
    if x <= 1: f1x = -x
    elif x <=3: f1x = x - 2
    elif x <= 4: f1x = 4 - x
    else: f1x = x-4
    f2x = (x - 5)**2
    
    return f1x, f2x


pop = Population(func, num_objectives = 2, num_variables = 1, num_particles = 10)

pop.initialize_population(-5, 10)

rep = Repository(repository_size = 200)
PSO = Optim(personal_learn_coeff = 1, global_learn_coeff = 2, max_iterations = 100, mutation_rate=0.1)

PSO.solve(pop, rep)

