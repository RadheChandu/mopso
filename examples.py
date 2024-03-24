from mopso import *



# func = lambda x: (1/x[0], x[0]**2 - x[1])

# pop = Population(func, 2, 2, 20)
# pop.initialize_population([0.1, 0.5], [1, 2])

# rep = Repository(repositorySize = 50)
# PSO = Optim(personalLearnCoeff = 0.2, globalLearnCoeff = 0.4, maxIterations = 50)

# PSO.solve(pop, rep)


# func = lambda x: (x**2, (x - 2)**2)

# pop = Population(func, 2, 1, 10)

# pop.initialize_population(-100000, 100000)

# rep = Repository(repositorySize = 100)
# PSO = Optim(personalLearnCoeff = 0.1, globalLearnCoeff = 0.2, maxIterations = 100)

# PSO.solve(pop, rep)

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

