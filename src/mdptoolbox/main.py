import example
import mdp

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

# Function to create a masked array
def masked_arr(n_iter, max_iter):
    arr = ma.empty((n_iter, max_iter))
    arr.mask = True
    return arr

# Function to create a list of masked arrays
def create_arr(n_iter, max_iter, n_exp=1):
    return [masked_arr(n_iter, max_iter) for _ in range(n_exp)]


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def last_iter_mskd(msk):
    falses = np.where(~msk)
    id = [np.searchsorted(falses[0], i, side='right')-1 for i in range(np.shape(msk)[0])]
    idy = [falses[0][id[i]] for i in range(np.shape(msk)[0])]
    return idy

# Function that will apply policy iteration. l should be a list. 
def runPI(n_iter, P, R, gamma, mode, max_iter, exampleName, lambdas=[1], fill_value=-99999):
    # n_iter: number of repetitions of the algorithm
    # P and R: transition probabilities; reward matrix
    # gamma: discout value 0<=gamma<=1
    # mode: 0, 1, 2 (maximizing, satisficing, satisficing with minimal variance)
    # max_iter: int, specificing number of iterations in algorihm
    # lambdas: list of floats, with every value between 0 and 1
    # fill_value: large value that is different form all output
    # exampleName: str, name of example

    arr_list = create_arr(n_iter, max_iter, len(lambdas))
    for i in range(n_iter):
        for j in range(len(lambdas)):
            print(j)
            PI = mdp.PolicyIteration(transitions=P, reward=R, discount=gamma, mode = mode, plot = True, l = lambdas[j], max_iter = max_iter)
            PI.run()
            arr_list[j][i, 0:len(PI.vlist)] = PI.vlist
    
    for j in range(len(lambdas)):
        fileName = str('PI_') + str(exampleName) + str('_l=') + str(lambdas[j]) + str('_n_iter=') + str(n_iter) + str('_max_iter=') + str(max_iter) + str('_fill_value=') + str(fill_value) + str('.csv')
        np.savetxt("src/mdptoolbox/saves/" + fileName, arr_list[j].filled(fill_value=fill_value), delimiter = ",")
        
# n_iter = 100
# max_iter = 1000
# P, R, _ = example.smallMDP()
# gamma = 0.5
# mode = 1
# lambdas = [0, 0.25, 0.5, 0.75, 1]
# # fill_value = -99999
# exampleName = 'smallMDP'


# runPI(n_iter, P, R, gamma, mode, max_iter, exampleName, lambdas)

# Function that will apply Q learning. l should be a list. 
def runQ(n_iter, P, R, gamma, mode, max_iter, exampleName, lambdas=[1], fill_value=-99999):
    # n_iter: number of repetitions of the algorithm
    # P and R: transition probabilities; reward matrix
    # gamma: discout value 0<=gamma<=1
    # mode: 0, 1, 2 (maximizing, satisficing, satisficing with minimal variance)
    # max_iter: int, specificing number of iterations in algorihm
    # lambdas: list of floats, with every value between 0 and 1
    # fill_value: large value that is different form all output
    # exampleName: str, name of example

    arr_list = create_arr(n_iter, max_iter, len(lambdas))
    for i in range(n_iter):
        for j in range(len(arr_list)):
            Q = mdp.QLearningModified(P, R, gamma, mode = mode, plot = True, l = lambdas[j], n_iter = max_iter)
            Q.run()
            arr_list[j][i, 0:len(Q.vlist)] = Q.vlist
    
    for j in range(len(arr_list)):
        fileName = str('Q_') + str(exampleName) + str('_mode=') + str(mode) + str('_l=') + str(lambdas[j]) + str('_n_iter=') + str(n_iter) + str('_max_iter=') + str(max_iter) + str('_fill_value=') + str(fill_value) + str('.csv')
        np.savetxt("src/mdptoolbox/saves/" + fileName, arr_list[j].filled(fill_value=fill_value), delimiter = ",")

n_iter = 1
max_iter = 10000
P, R, _ = example.smallMDP()
gamma = 0.5
mode = 2
lambdas = [1]
# fill_value = -99999
exampleName = 'smallMDP'


runQ(n_iter, P, R, gamma, mode, max_iter, exampleName, lambdas)
