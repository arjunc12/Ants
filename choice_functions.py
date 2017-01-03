import numpy as np
import networkx as nx
from numpy.random import random, choice
MIN_DETECTABLE_PHEROMONE = 0
MIN_PHEROMONE = 0
PHEROMONE_THRESHOLD = 0
from collections import defaultdict

def next_edge_uniform(G, start, explore_prob, candidates=None):
    if candidates == None:
        candidates = G.neighbors(start)
    
    total_wt = 0.0
    explored = []
    unexplored = []
    explored_weights = []
    for candidate in candidates:
        wt = G[start][candidate]['weight']
        if wt <= PHEROMONE_THRESHOLD:
            unexplored.append(candidate)
        else:
            explored.append(candidate)
            explored_weights.append(wt)
            total_wt += wt
    flip = random()
    if (flip < explore_prob and len(unexplored) > 0) or (len(explored) == 0):
        next = choice(len(unexplored))
        next = unexplored[next]
        return next, True 
    else:
        explored_weights = np.array(explored_weights)
        explored_weights /= total_wt
        next = explored[choice(len(explored), 1, p=explored_weights)[0]]
        return next, False
    
def next_edge_max(G, start, explore_prob, candidates=None):
    if candidates == None:
        candidates = G.neighbors(start)
        
    max_wt = float("-inf")
    for candidate in candidates:
        max_wt = max(max_wt, G[start][candidate]['weight'])
    
    max_neighbors = []
    nonmax_neighbors = []
    for candidate in candidates:
        wt = G[start][candidate]['weight']
        if wt == max_wt and wt > PHEROMONE_THRESHOLD:
            max_neighbors.append(candidate)
        else:
            nonmax_neighbors.append(candidate)
            
    flip = random()
    if (flip < explore_prob and len(nonmax_neighbors) > 0) or (len(max_neighbors) == 0):
        next = choice(len(nonmax_neighbors))
        next = nonmax_neighbors[next]
        return next, True
    else:
        next = choice(len(max_neighbors))
        next = max_neighbors[next]
        return next, False
        
def next_edge_maxz(G, start, explore_prob, candidates=None):
    if candidates == None:
        candidates = G.neighbors(start)
        
    max_wt = float("-inf")
    for candidate in candidates:
        max_wt = max(max_wt, G[start][candidate]['weight'])
    
    max_neighbors = []
    nonmax_neighbors = []
    for candidate in candidates:
        wt = G[start][candidate]['weight']
        if wt == max_wt and wt > PHEROMONE_THRESHOLD:
            max_neighbors.append(candidate)
        elif wt <= PHEROMONE_THRESHOLD:
            nonmax_neighbors.append(candidate)
            
    flip = random()
    if (flip < explore_prob and len(nonmax_neighbors) > 0) or (len(max_neighbors) == 0):
        next = choice(len(nonmax_neighbors))
        next = nonmax_neighbors[next]
        return next, True
    else:
        next = choice(len(max_neighbors))
        next = max_neighbors[next]
        return next, False
        
def next_edge_rank(G, start, explore_prob, candidates=None):
    if candidates == None:
        candidates = G.neighbors(start)
        
    weights = defaultdict(list)
    for candidate in candidates:
        wt = G[start][candidate]['weight']
        weights[wt].append(candidate)
        
    rank_weights = list(reversed(sorted(weights.keys())))
    i = 0
    done = False
    next = None
    ex = False
    while not done:
        flip = random()
        if (flip <= 1 - explore_prob) or (i == len(rank_weights) - 1):
            rank_wt = rank_weights[i]
            rank_neighbors = weights[rank_wt]
            next = choice(len(rank_neighbors))
            next = rank_neighbors[next]
            done = True
        else:
            i += 1
            ex = True
    
    if G[start][next]['weight'] == MIN_PHEROMONE:
        ex = True
            
    return next, ex            
    
def next_edge(G, start, explore_prob, strategy='uniform', prev=None, dest=None, \
              search=True, backtrack=False):
    candidates = G.neighbors(start)
    if (dest != None) and (dest in candidates):
        return dest, False
    if candidates == [prev]:
        return prev, False
    elif type(prev) == list and candidates == prev:
        next = choice(len(prev))
        next = prev[next]
        return next, False
    if (prev != None) and (not backtrack):
        if type(prev) != list:
            prev = [prev]
        for node in prev:
            #assert prev in candidates
            if node in candidates:
                candidates.remove(node)
    choice_func = None
    if (strategy == 'uniform') or (strategy in ['hybrid', 'hybridz'] and search):
        choice_func = next_edge_uniform
    elif (strategy == 'max') or (strategy == 'hybrid' and not search):
        choice_func = next_edge_max
    elif (strategy == 'maxz') or (strategy == 'hybridz' and not search):
        choice_func = next_edge_maxz
    elif strategy == 'rank':
        choice_func = next_edge_rank
    return choice_func(G, start, explore_prob, candidates)
    
def uniform_likelihood(G, source, dest, explore, prev=None):
    chosen_wt = G[source][dest]['weight']
    total = 0.0
    explored = 0
    unexplored = 0
    neighbors = G.neighbors(source)
    if prev != None:
        assert prev in neighbors
        neighbors.remove(prev)
    for n in neighbors:
        wt = G[source][n]['weight']
        assert wt >= MIN_PHEROMONE
        if wt <= MIN_DETECTABLE_PHEROMONE:
            unexplored += 1
        else:
            explored += 1
            total += wt
    assert explored + unexplored == len(neighbors)
    if explored == 0:
        assert unexplored == len(neighbors)
        return 1.0 / unexplored
    elif chosen_wt <= MIN_DETECTABLE_PHEROMONE:
        return explore * (1.0 / unexplored)
    else:
        prob = chosen_wt / total
        prob *= (1 - explore)
        return prob
             
def max_edge_likelihood(G, source, dest, explore, prev=None):
    max_wt = MIN_PHEROMONE
    max_neighbors = []
    neighbors = G.neighbors(source)
    weights = []
    if prev != None:
        assert prev in neighbors
        neighbors.remove(prev)
    chosen_wt = G[source][dest]['weight']
    for n in neighbors:
        wt = G[source][n]['weight']
        assert wt >= MIN_PHEROMONE
        max_wt = max(wt, max_wt)
        weights.append(wt)
    
    explored = 0
    unexplored = 0
    for n in neighbors:
        wt = G[source][n]['weight']
        if wt == max_wt and wt > MIN_DETECTABLE_PHEROMONE:
            explored += 1
        else:
            unexplored += 1
    
    if explored == 0:
        assert unexplored == len(neighbors)
        assert MIN_PHEROMONE <= chosen_wt <= MIN_DETECTABLE_PHEROMONE
        #return 1.0 / unexplored
        return 1 - explore
        
    if chosen_wt == max_wt:
        prob = 1.0 / explored
        prob *= (1 - explore)
        return prob
    else:
        assert chosen_wt < max_wt
        assert unexplored > 0
        return explore * (1.0 / unexplored)
        
def maxz_edge_likelihood(G, source, dest, explore, prev=None):
    chosen_wt = G[source][dest]['weight']
    max_wt = MIN_PHEROMONE
    max_neighbors = []
    zero_neighbors = []
    neighbors = G.neighbors(source)
    if prev != None:
        assert prev in neighbors
        neighbors.remove(prev)
    for n in neighbors:
        wt = G[source][n]['weight']
        assert wt >= MIN_PHEROMONE
        if wt <= MIN_DETECTABLE_PHEROMONE:
            zero_neighbors.append(n)
        else:
            if wt > max_wt:
                max_wt = wt
                max_neighbors = [n]
            elif wt == max_wt:
                max_neighbors.append(n)

    if dest in max_neighbors:
        assert max_wt > MIN_DETECTABLE_PHEROMONE
        assert chosen_wt == max_wt
        prob = 1.0 / len(max_neighbors)
        if len(zero_neighbors) > 0:
            prob *= (1 - explore)
        return prob
    elif dest in zero_neighbors:
        assert MIN_PHEROMONE <= chosen_wt <= MIN_DETECTABLE_PHEROMONE
        return explore / (len(zero_neighbors))
    else:
        assert MIN_DETECTABLE_PHEROMONE < chosen_wt < max_wt
        return 0
        
def rank_likelihood(G, source, dest, explore, prev=None):
    chosen_wt = G[source][dest]['weight']
    weights = defaultdict(list)
    neighbors = G.neighbors(source)
    if prev != None:
        assert prev in neighbors
        neighbors.remove(prev)
    for n in neighbors:
        wt = G[source][n]['weight']
        weights[wt].append(n)
        
    rank_weights = list(reversed(sorted(weights.keys())))
    assert chosen_wt in rank_weights
    prob = 1.0
    for i in xrange(len(rank_weights)):
        wt = rank_weights[i]
        if chosen_wt == wt:
            prob /= len(weights[wt])
            prob *= (1 - explore)
            return prob
        else:
            prob *= explore
    
def choice_prob(G, source, dest, explore_prob, prev=None, strategy='uniform'):
    likelihood_func = None
    if strategy == 'uniform':
        likelihood_func = uniform_likelihood
    elif strategy in ['max', 'hybrid']:
        likelihood_func = max_edge_likelihood
    elif strategy in ['maxz', 'hybridz']:
        likelihood_func = maxz_edge_likelihood
    elif strategy == 'rank':
        likelihood_func = rank_likelihood
    else:
        raise ValueError('invalid strategy')
    return likelihood_func(G, source, dest, explore_prob, prev)