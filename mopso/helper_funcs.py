
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize(costs):
    scaler = MinMaxScaler()
    return scaler.fit_transform(costs)      
        

def get_leader_using_MDPL(repObject, popObject):
    
    epsilon = 1e-6
    if repObject.num_particles > 0:        
        popCosts = popObject.cost
        repCosts = repObject.cost
        
        normCosts = normalize(np.vstack((popCosts, repCosts)))
        
        popCosts = normCosts[:popCosts.shape[0], :]
        repCosts = normCosts[popCosts.shape[0]:, :]
        index = []
        for i in popCosts:
            distance = abs(repCosts[:,1]*i[0] - repCosts[:,0]*i[1])/(np.linalg.norm(repCosts, axis = 1) + epsilon)
            index.append(np.argmin(distance))
        return repObject.position[index,:]
    else:
        return popObject.position

def dominates(x, y):
    if len(y.shape) == 1:
        return np.all(x <= y) & np.any(x < y)
    else:
        return np.all(x <= y, axis = 1) & np.any(x < y, axis = 1)

def get_domination_status(costs):
    num_particles = costs.shape[0]
    dom_status = np.ones(costs.shape[0], dtype = bool)

    for i in range(num_particles):
        if dom_status[i]:
            dominion = dominates(costs[i,:], costs)
            dom_status[dominion] = False
    return dom_status