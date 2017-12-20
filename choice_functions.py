import numpy as np
import networkx as nx
from numpy.random import random, choice

# minimum amount of pheromone that needs to be on an edge for an ant to detect it
# zero-edges are defined to be edges with less pheromone than the minimum detectable
MIN_DETECTABLE_PHEROMONE = 0
# minimum amount of pheromone that can be on an edge
MIN_PHEROMONE = 0

DBERG_OFFSET = 1

from collections import defaultdict

STRATEGY_CHOICES = ['uniform', 'max', 'hybrid', 'maxz', 'hybridz', 'rank', 'hybridm',\
                    'hybridr', 'ranku', 'uniform2', 'max2', 'maxu', 'maxa', 'ranka',\
                    'unweighted', 'dberg', 'rankt']

def local_graph(G, start):
    G2 = nx.Graph()
    for neighbor in G.neighbors(start):
        G2.add_edge(start, neighbor)
        G2[start][neighbor]['weight'] = G[start][neighbor]['weight']
        
    return G2

def next_edge_unweighted(G, start, explore_prob=None, candidates=None):
    if candidates == None:
        candidates = G.neighbors(start)

    next = choice(len(candidates))
    next = next[choice]
    return next, False

def next_edge_uniform(G, start, explore_prob, candidates=None):
    '''
    Chooses next edge randomly according to the uniform model.  With some probability
    it picks equally among the edges with 0 weight (or weight less than the detectable
    threshold).  Otherwise it picks among the edges of non-zero weight with probability
    proportional to the edge weights.
    '''
    if candidates == None:
        candidates = G.neighbors(start)
    
    total_wt = 0.0
    explored = []
    unexplored = []
    explored_weights = []
    
    # separate zero and non-zero neighbors, count the total weight
    for candidate in candidates:
        wt = G[start][candidate]['weight']
        if wt <= MIN_DETECTABLE_PHEROMONE:
            unexplored.append(candidate)
        else:
            explored.append(candidate)
            explored_weights.append(wt)
            total_wt += wt
            
    flip = random()
    # pick zero neighbor with probability explore_prob, or if all edges are zero edges
    if (flip < explore_prob and len(unexplored) > 0) or (len(explored) == 0):
        next = choice(len(unexplored))
        next = unexplored[next]
        return next, True 
    else:
        explored_weights = np.array(explored_weights)
        explored_weights /= total_wt
        next = explored[choice(len(explored), 1, p=explored_weights)[0]]
        return next, False

def next_edge_uniform2(G, start, explore_prob, candidates=None):
    '''
    Picks edges with probability proportional to the square of the edge weights
    '''
    return next_edge_uniformn(G, start, explore_prob, 2, candidates)

def next_edge_uniformn(G, start, explore_prob, n, candidates=None):
    '''
    Picks edges with probability proportional to the edge weights raised to the n-th power
    '''
    if candidates == None:
        candidates = G.neighbors(start)
    
    total_wt = 0.0
    explored = []
    unexplored = []
    explored_weights = []
    for candidate in candidates:
        wt = G[start][candidate]['weight']
        if (wt ** n) <= MIN_DETECTABLE_PHEROMONE:
            unexplored.append(candidate)
        else:
            explored.append(candidate)
            explored_weights.append(wt ** n)
            total_wt += wt ** n
    flip = random()
    if (flip < explore_prob and len(unexplored) > 0) or (len(explored) == 0):
        next = choice(len(unexplored))
        next = unexplored[next]
        return next, True
    elif total_wt == 0:
        print explored_weights
        next = choice(len(candidates))
        next = candidates[next]
        return next, True
    else:
        explored_weights = np.array(explored_weights)
        explored_weights /= total_wt
        next = explored[choice(len(explored), 1, p=explored_weights)[0]]
        return next, False

def next_edge_dberg(G, start, explore_prob, candidates=None, offset=DBERG_OFFSET):
    if explore_prob == 0:
        return next_edge_max(G, start, explore_prob, candidates)
        
    G2 = local_graph(G, start)
    for neighbor in G2.neighbors(start):
        wt = G2[start][neighbor]['weight']
        wt += offset
        
    a = 1 / explore_prob
    assert 0 <= a <= 1
    return next_edge_uniformn(G2, start, explore_prob, a, candidates)
    
def next_edge_max(G, start, explore_prob, candidates=None):
    '''
    With some probability, picks equally among the edges whose weight is lower than the
    highest weighted adjacent edge.  Otherwise, picks equally among all edges tied for the
    highest edge weight.
    '''
    if candidates == None:
        candidates = G.neighbors(start)
    
    # compute highest adjacent edge weight    
    max_wt = float("-inf")
    for candidate in candidates:
        max_wt = max(max_wt, G[start][candidate]['weight'])
    
    # split neighbors into maximally weighted and non-maximally weighted edges
    max_neighbors = []
    nonmax_neighbors = []
    for candidate in candidates:
        wt = G[start][candidate]['weight']
        # Edges with too small weight not considered maximal
        if wt == max_wt and wt > MIN_DETECTABLE_PHEROMONE:
            max_neighbors.append(candidate)
        else:
            nonmax_neighbors.append(candidate)
            
    flip = random()
    # Explores non-maximal edge with probability explore_prob
    if (flip < explore_prob and len(nonmax_neighbors) > 0) or (len(max_neighbors) == 0):
        next = choice(len(nonmax_neighbors))
        next = nonmax_neighbors[next]
        return next, True
    else:
        next = choice(len(max_neighbors))
        next = max_neighbors[next]
        return next, False

def next_edge_max2(G, start, explore_prob, candidates=None):
    if candidates == None:
        candidates = G.neighbors(start)

    max_wt = float("-inf")
    for candidate in candidates:
        max_wt = max(max_wt, G[start][candidate]['weight'])

    max_neighbors = []
    mid_neighbors = []
    lower_neighbors = []

    for candidate in candidates:
        wt = G[start][candidate]['weight']
        if wt > MIN_DETECTABLE_PHEROMONE:
            if wt == max_wt:
                max_neighbors.append(candidate)
            else:
                mid_neighbors.append(candidate)
        else:
            lower_neighbors.append(candidate)

    flip1 = random()
    flip2 = random()

    maxn = len(max_neighbors)
    midn = len(mid_neighbors)
    lowern = len(lower_neighbors)

    if midn > 0:
        assert maxn > 0

    if (flip1 > explore_prob and maxn > 0) or (midn + lowern == 0):
        next = choice(maxn)
        next = max_neighbors[next]
        return next, False
    elif (flip2 > explore_prob and midn > 0) or lowern == 0:
        next = choice(midn)
        next = mid_neighbors[next]
        return next, True
    else:
        next = choice(lowern)
        next = lower_neighbors[next]
        return next, True
       
def next_edge_maxu(G, start, explore_prob, candidates=None):
    if candidates == None:
        candidates = G.neighbors(start)
        
    weights = []
    for candidate in candidates:
        weights.append(G[start][candidate]['weight'])
        
    max_wt = max(weights)
    top = []
    bottom = []
    for candidate in candidates:
        if G[start][candidate]['weight'] == max_wt:
            top.append(candidate)
        else:
            bottom.append(candidate)
            
    flip = random()
    if (flip <= 1 - explore_prob) or (len(bottom) == 0):
        next = choice(len(top))
        next = top[next]
        return next, False
    else:
        next, ex = next_edge_uniform(G, start, explore_prob, candidates=bottom)
        return next, True


def next_edge_maxa(G, start, explore_prob, candidates=None):
    '''
    With some probability, picks equally among the edges whose weight is lower than the 
    Otherwise, picks equally among all edges tied for the highest edge weight. Note
    that on explore steps it can still pick the highest-weighted edge
    '''
    if candidates == None:
        candidates = G.neighbors(start)
    
    # compute highest adjacent edge weight    
    max_wt = float("-inf")
    for candidate in candidates:
        max_wt = max(max_wt, G[start][candidate]['weight'])
    
    # split neighbors into maximally weighted and non-maximally weighted edges
    max_neighbors = []
    nonmax_neighbors = []
    for candidate in candidates:
        wt = G[start][candidate]['weight']
        # Edges with too small weight not considered maximal
        if wt == max_wt and wt > MIN_DETECTABLE_PHEROMONE:
            max_neighbors.append(candidate)
        else:
            nonmax_neighbors.append(candidate)
            
    flip = random()
    # Explores non-maximal edge with probability explore_prob
    if (flip < explore_prob and len(nonmax_neighbors) > 0) or (len(max_neighbors) == 0):
        next = choice(len(candidates))
        next = candidates[next]
        return next, True
    else:
        next = choice(len(max_neighbors))
        next = max_neighbors[next]
        return next, False

def next_edge_maxz(G, start, explore_prob, candidates=None):
    '''
    With some probability, picks equally among zero edges, otherwise picks equally among
    maximal edges. This choice function ignores all edges in the 'middle', i.e. edges that
    are neither maximal nor minimal
    '''
    if candidates == None:
        candidates = G.neighbors(start)
        
    max_wt = float("-inf")
    for candidate in candidates:
        max_wt = max(max_wt, G[start][candidate]['weight'])
    
    max_neighbors = []
    nonmax_neighbors = []
    for candidate in candidates:
        wt = G[start][candidate]['weight']
        if wt == max_wt and wt > MIN_DETECTABLE_PHEROMONE:
            max_neighbors.append(candidate)
        elif wt <= MIN_DETECTABLE_PHEROMONE:
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
        if wt <= MIN_DETECTABLE_PHEROMONE:
            wt = 0
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
 
def next_edge_ranku(G, start, explore_prob, candidates=None):
    if candidates == None:
        candidates = G.neighbors(start)
        
    weights = []
    for candidate in candidates:
        weights.append(G[start][candidate]['weight'])
        
    max_wt = max(weights)
    top = []
    bottom = []
    for candidate in candidates:
        if G[start][candidate]['weight'] == max_wt:
            top.append(candidate)
        else:
            bottom.append(candidate)
            
    flip = random()
    if (flip <= 1 - explore_prob) or (len(bottom) == 0):
        next = choice(len(top))
        next = top[next]
        return next, False
    else:
        next, ex = next_edge_uniform(G, start, explore_prob, candidates=bottom)
        return next, ex
    
def next_edge_ranka(G, start, explore_prob, candidates=None):
    if candidates == None:
        candidates = G.neighbors(start)
    
    weights = defaultdict(list)
    for candidate in candidates:
        wt = G[start][candidate]['weight']
        if wt <= MIN_DETECTABLE_PHEROMONE:
            wt = 0
        weights[wt].append(candidate)
        
    rank_weights = list(reversed(sorted(weights.keys())))
    i = 0
    done = False
    next = None
    ex = False
    rank_neighbors = []
    while not done:
        flip = random()
        if (flip <= 1 - explore_prob) or (i == len(rank_weights) - 1):
            rank_wt = rank_weights[i]
            rank_neighbors += weights[rank_wt]
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
    #print start, candidates, prev
    for candidate in candidates[:]:
        if G[start][candidate]['anti_pheromone'] > MIN_DETECTABLE_PHEROMONE:
            candidates.remove(candidate)
    if len(candidates) == 0:
        candidates = G.neighbors(start)
    if (dest != None) and (dest in candidates):
        return dest, False
    if candidates == [prev]:
        return prev, False
    elif type(prev) == list and sorted(candidates) == sorted(prev):
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
    assert len(candidates) > 0
    
    if len(candidates) == 1:
        return candidates[0], False
    
    choice_func = None
    if (strategy == 'uniform') or (strategy in ['hybrid', 'hybridz', 'hybridm', 'hybridr'] and search):
        choice_func = next_edge_uniform
    elif (strategy == 'max') or (strategy in ['hybrid', 'hybridm'] and not search):
        choice_func = next_edge_max
    elif (strategy == 'maxz') or (strategy == 'hybridz' and not search):
        choice_func = next_edge_maxz
    elif (strategy == 'rank') or (strategy == 'hybridr' and not search):
        choice_func = next_edge_rank
    elif (strategy == 'ranku'):
        choice_func = next_edge_ranku
    elif (strategy == 'uniform2'):
        choice_func = next_edge_uniform2
    elif strategy == 'max2':
        choice_func = next_edge_max2
    elif strategy == 'maxu':
        choice_func = next_edge_maxu
    elif strategy == 'maxa':
        choice_func = next_edge_maxa
    elif strategy == 'ranka':
        choice_func = next_edge_ranka
    elif strategy == 'unweighted':
        choice_func = next_edge_unweighted
    elif strategy == 'dberg':
        choice_func = next_edge_dberg
    else:
        raise ValueError('invalid strategy')
    return choice_func(G, start, explore_prob, candidates)

def unweighted_likelihood(G, source, dest, explore=None, prev=None):
    candidates = G.neighbors(source)
    if prev != None:
        assert prev in candidates
        candidates.remove(prev)
    return 1.0 / len(candidates)

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
        if unexplored > 0:
            prob *= (1 - explore)
        return prob
        
def uniformn_likelihood(G, source, dest, explore, n, prev=None):
    chosen_wt = G[source][dest]['weight'] ** n
    total = 0.0
    explored = 0
    unexplored = 0
    neighbors = G.neighbors(source)
    if prev != None:
        assert prev in neighbors
        neighbors.remove(prev)
    for neighbor in neighbors:
        wt = G[source][neighbor]['weight'] ** n
        assert wt >= MIN_PHEROMONE
        if wt <= MIN_DETECTABLE_PHEROMONE:
            unexplored += 1
        else:
            explored += 1
            total += wt
    assert explored + unexplored == len(neighbors)
    if explored == 0 or total <= 0:
        assert unexplored == len(neighbors)
        return 1.0 / unexplored
    elif chosen_wt <= MIN_DETECTABLE_PHEROMONE:
        return explore * (1.0 / unexplored)
    else:
        assert total > 0
        prob = chosen_wt / total
        if unexplored > 0:
            prob *= (1 - explore)
        return prob
        
def uniform2_likelihood(G, source, dest, explore, prev=None):
    return uniformn_likelihood(G, source, dest, explore, 2, prev)
    
def dberg_likelihood(G, source, dest, explore, prev=None, offset=DBERG_OFFSET):
    if explore == 0:
        return max_edge_likelihood(G, source, dest, explore, prev)
    G2 = local_graph(G, source)
    for u, v in G2.edges_iter():
        G2[u][v]['weight'] += offset
    a = 1 / explore
    return uniformn_likelihood(G2, source, dest, explore, a, prev)
             
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
        return 1.0 / unexplored
        
    if chosen_wt == max_wt:
        prob = 1.0 / explored
        if unexplored > 0:
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
        if unexplored > 0:
            prob *= (1 - explore)
        return prob
    elif dest in zero_neighbors:
        assert MIN_PHEROMONE <= chosen_wt <= MIN_DETECTABLE_PHEROMONE
        return explore / (len(zero_neighbors))
    else:
        assert MIN_DETECTABLE_PHEROMONE < chosen_wt < max_wt
        return 0

def max2_likelihood(G, source, dest, explore, prev=None):
    maxn = 0
    midn = 0
    lowern = 0

    max_wt = float("-inf")
    candidates = G.neighbors(source)
    if prev != None:
        assert prev in candidates
        candidates.remove(prev)

    for candidate in candidates:
        wt = G[source][candidate]['weight']
        max_wt = max(wt, max_wt)

    for candidate in candidates:
        wt = G[source][candidate]['weight']
        if wt <= MIN_DETECTABLE_PHEROMONE:
            lowern += 1
        elif wt == max_wt:
            maxn += 1
        else:
            midn += 1

    chosen_wt = G[source][dest]['weight']
    if chosen_wt <= MIN_DETECTABLE_PHEROMONE:
        assert lowern > 0
        prob = 1.0 / lowern
        if maxn + midn > 0:
            prob *= explore ** 2
        return prob
    elif chosen_wt == max_wt:
        assert maxn > 0
        prob = 1.0 / maxn
        if lowern + midn > 0:
            prob *= 1 - explore
        return prob
    else:
        assert midn > 0
        prob = 1.0 / midn
        assert maxn > 0
        prob *= explore
        if lowern > 0:
            prob *= (1 - explore)
        return prob  

def maxu_likelihood(G, source, dest, explore, prev=None):
    chosen_wt = G[source][dest]['weight']
    max_wt = float('-inf')
    total_wt = 0.0
    candidates = G.neighbors(source)
    if prev != None:
        assert prev in candidates
        candidates.remove(prev)
        
    for candidate in candidates:
        wt = G[source][candidate]['weight']
        max_wt = max(max_wt, wt)
    
    upper_neighbors = 0
    mid_neighbors = 0
    zero_neighbors = 0
    for candidate in candidates:
        wt = G[source][candidate]['weight']
        if wt <= MIN_DETECTABLE_PHEROMONE:
            zero_neighbors += 1
        elif wt == max_wt:
            upper_neighbors += 1
        else:
            total_wt += wt
            mid_neighbors += 1
    
    lower_neighbors = zero_neighbors + mid_neighbors
    if chosen_wt == max_wt and max_wt > MIN_DETECTABLE_PHEROMONE:
        assert upper_neighbors > 0
        prob = 1.0 / upper_neighbors
        if lower_neighbors > 0:
            prob *= (1 - explore)
        return prob
    elif chosen_wt > MIN_DETECTABLE_PHEROMONE:
        assert chosen_wt < max_wt
        assert total_wt > 0
        prob = chosen_wt / total_wt
        prob *= explore
        if zero_neighbors > 0:
            prob *= (1 - explore)
        return prob
    else:
        assert chosen_wt <= MIN_DETECTABLE_PHEROMONE
        assert lower_neighbors > 0
        prob = 1.0 / lower_neighbors
        if upper_neighbors > 0:
            prob *= explore
        if mid_neighbors > 0:
            prob *= explore
        return prob

def maxa_likelihood(G, source, dest, explore, prev=None):
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
        return 1.0 / unexplored
        
    if chosen_wt == max_wt:
        assert explored > 0
        prob = 1.0 / explored
        if unexplored > 0:
            prob *= (1 - explore)
            prob += explore * (1.0 / len(neighbors))
        return prob
    else:
        assert chosen_wt < max_wt
        assert unexplored > 0
        return explore * (1.0 / unexplored)

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
            if i < len(rank_weights) - 1:
                prob *= (1 - explore)
            return prob
        else:
            prob *= explore
            
def ranku_likelihood(G, source, dest, explore, prev=None):
    chosen_wt = G[source][dest]['weight']
    max_wt = float('-inf')
    total_wt = 0.0
    candidates = G.neighbors(source)
    if prev != None:
        assert prev in candidates
        candidates.remove(prev)
        
    for candidate in candidates:
        wt = G[source][candidate]['weight']
        max_wt = max(max_wt, wt)
    
    upper_neighbors = 0
    mid_neighbors = 0
    zero_neighbors = 0
    for candidate in candidates:
        wt = G[source][candidate]['weight']
        if wt <= MIN_DETECTABLE_PHEROMONE:
            zero_neighbors += 1
        elif wt == max_wt:
            upper_neighbors += 1
        else:
            total_wt += wt
            mid_neighbors += 1
    
    lower_neighbors = zero_neighbors + mid_neighbors
    if chosen_wt == max_wt and max_wt > MIN_DETECTABLE_PHEROMONE:
        assert upper_neighbors > 0
        prob = 1.0 / upper_neighbors
        if lower_neighbors > 0:
            prob *= (1 - explore)
        return prob
    elif chosen_wt > MIN_DETECTABLE_PHEROMONE:
        assert chosen_wt < max_wt
        assert total_wt > 0
        prob = chosen_wt / total_wt
        prob *= explore
        if zero_neighbors > 0:
            prob *= (1 - explore)
        return prob
    else:
        assert chosen_wt <= MIN_DETECTABLE_PHEROMONE
        assert lower_neighbors > 0
        prob = 1.0 / lower_neighbors
        if upper_neighbors > 0:
            prob *= explore
        if mid_neighbors > 0:
            prob *= explore
        return prob

def ranka_likelihood(G, source, dest, explore, prev=None):
    chosen_wt = G[source][dest]['weight']
    if chosen_wt <= MIN_DETECTABLE_PHEROMONE:
        chosen_wt = 0
    weights = defaultdict(list)
    neighbors = G.neighbors(source)
    if prev != None:
        assert prev in neighbors
        neighbors.remove(prev)
    for n in neighbors:
        wt = G[source][n]['weight']
        if wt <= MIN_DETECTABLE_PHEROMONE:
            wt = 0
        weights[wt].append(n)
        
    rank_weights = list(reversed(sorted(weights.keys())))
    assert chosen_wt in rank_weights
    prob = 1.0
    rank_neighbors = 0
    for i in xrange(len(rank_weights)):
        wt = rank_weights[i]
        if chosen_wt == wt:
            rank_neighbors += len(weights[wt])
            prob /= rank_neighbors
            if i < len(rank_weights) - 1:
                prob *= (1 - explore)
            return prob
        else:
            prob *= explore

def rankt_likelihood(G, source, dest, explore, prev=None):
    chosen_wt = G[source][dest]['weight']
    if chosen_wt <= MIN_DETECTABLE_PHEROMONE:
        chosen_wt = 0

    weights = defaultdict(list)
    neighbors = G.neighbors(source)
    if prev != None:
        assert prev in neighbors

    for n in neighbors:
        wt = G[source][n]['weight']
        if wt <= MIN_DETECTABLE_PHEROMONE:
            wt = 0
        weights[wt].append(n)
    
    ranked_weights = list(reversed(sorted(weights.keys())))
    prob = 1.0
    for i in xrange(len(ranked_weights)):
        wt = ranked_weights[i]
        rank_neighbors = weights[wt]
        n = len(rank_neighbors)
        if chosen_wt == wt:
            sub_prob = 0
            for j in xrange(1, n + 1):
                p = (1.0 / n) * (explore ** (j - 1))
                if not (i == len(ranked_weights) - 1 and j == n):
                    p *= (1 - explore)
                sub_prob += p
            prob *= sub_prob
            return prob
        else:
            prob *= (explore ** n)

def get_likelihood_func(strategy):
    likelihood_func = None
    if strategy == 'uniform':
        likelihood_func = uniform_likelihood
    elif strategy in ['max', 'hybrid']:
        likelihood_func = max_edge_likelihood
    elif strategy in ['maxz', 'hybridz']:
        likelihood_func = maxz_edge_likelihood
    elif strategy in ['rank', 'hybridr']:
        likelihood_func = rank_likelihood
    elif strategy == 'ranku':
        likelihood_func = ranku_likelihood
    elif strategy == 'uniform2':
        likelihood_func = uniform2_likelihood
    elif strategy == 'max2':
        likelihood_func = max2_likelihood
    elif strategy == 'maxu':
        likelihood_func = maxu_likelihood
    elif strategy == 'maxa':
        likelihood_func = maxa_likelihood
    elif strategy == 'ranka':
        likelihood_func = ranka_likelihood
    elif strategy == 'unweighted':
        likelihood_func = unweighted_likelihood
    elif strategy == 'dberg':
        likelihood_func = dberg_likelihood
    elif strategy == 'rankt':
        likelihood_func = rankt_likelihood
    else:
        raise ValueError('invalid strategy')
    return likelihood_func
        
def choice_prob(G, source, dest, explore_prob, prev=None, strategy='uniform'):
    likelihood_func = get_likelihood_func(strategy)
    return likelihood_func(G, source, dest, explore_prob, prev)


# functions for determining if a choice counts as an explore step
    
def is_explore_uniform(G, source, dest, prev=None):
    has_unexplored = False
    candidates = G.neighbors(source)
    if prev != None:
        assert prev in candidates
        candidates.remove(prev)
    for candidate in candidates:
        wt = G[source][dest]['weight']
        if wt > MIN_DETECTABLE_PHEROMONE:
            has_unexplored = True
            break
            
    if has_unexplored:
        chosen_wt = G[source][dest]['weight']
        return chosen_wt <= MIN_DETECTABLE_PHEROMONE
    else:
        None
    
def is_explore_max(G, source, dest, prev=None):
    max_wt = None
    candidates = G.neighbors(source)
    if prev != None:
        assert prev in candidates
        candidates.remove(prev)
    for candidate in candidates:
        wt = G[source][candidate]['weight']
        if wt > MIN_DETECTABLE_PHEROMONE:
            if max_wt == None:
                max_wt = wt
            else:
                max_wt = max(wt, max_wt)
                
    if max_wt == None:
        return None
    else:
        chosen_wt = G[source][dest]['weight']
        return chosen_wt < max_wt
    
def is_explore(G, source, dest, strategy='rank', prev=None):
    explore_func = None
    if strategy == 'uniform':
        explore_func = is_explore_uniform
    else:
        explore_func = is_explore_max
    return explore_func(G, source, dest, prev)

def main():
    G = nx.Graph()
    G.add_edge('a', 'b')
    G.add_edge('a', 'c')
    G.add_edge('a', 'd')
    G['a']['b']['weight'] = 2.5
    G['a']['c']['weight'] = 2.5
    G['a']['d']['weight'] = 1
    explore_prob = 0.1
    for n in G.neighbors('a'):
        print '----'
        print n
        print 'rankt', rankt_likelihood(G, 'a', n, explore_prob, prev=None)
        print 'rank', rank_likelihood(G, 'a', n, explore_prob, prev=None)

if __name__ == '__main__':
    main()
