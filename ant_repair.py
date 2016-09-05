#!/usr/bin/env python

from __future__ import division
import networkx as nx
import time,logging
from optparse import OptionParser
import argparse
from matplotlib import pylab as PP
from numpy.random import seed,choice, random
from numpy import mean,median, array, argmax, where
from collections import defaultdict
import os
from matplotlib import animation
from scipy.stats import spearmanr
from scipy.stats import entropy

#seed(10301949)

Minv = {} # node tuple -> node id
M = {}    # node id -> node tuple

Ninv = {}    # edge -> edge id
N = {}       # edge id -> edge

#MAX_STEPS= 10000
MIN_PHEROMONE = 0
pos = {}
node_color,node_size = [],[]
edge_color,edge_width = [],[]
P = []
path_thickness = 1.5
pheromone_thickness = 1
ant_thickness = 25
DEBUG_PATHS = True
OUTPUT_GRAPHS = False

DEAD_END = False
BREAK = False
BACKTRACK = False
# EXPLORE_PROB1 = 0.00000001
# EXPLORE_PROB2 = 0.02
ADD_PRUNE = 0.1
MIN_ADD = 1

MAX = False
INIT_WEIGHT_FACTOR = 10
MAX_PATH_LENGTH = 20

""" Difference from tesht2 is that the ants go one at a time + other output variables. """ 

# PARAMETERS:
#   n=number of ants
#   delta=pheromone add-decay amount
#   choice function=max or probabilistic
#   trade-offs=speed of recovery vs. centrality
#   backwards-coming ants=ignore.
# TODO: add/decay as a function of n?
# TODO: output path edges taken by each ant to visualize.
# TODO: more memory of history?
# TODO: is [0-1] the right range? 

def init_graph(G):
    for i,u in enumerate(G.nodes_iter()):
        M[i] = u
        Minv[u] = i

    for u,v in G.edges_iter(): G[u][v]['weight'] = MIN_PHEROMONE
    
    for u in G.nodes():
        pos[u] = [u[0],u[1]] # position is the same as the label.
        

        if u in G.graph['nests']:
            node_size.append(100)
            node_color.append('r')
        else:
            node_size.append(10)
            node_color.append('k')

    for i, (u,v) in enumerate(G.edges()):
        Ninv[(u, v)] = i
        N[i] = (u, v)        
        Ninv[(v, u)] = i
        
        if (u, v) in P:
            edge_color.append('g')
            edge_width.append(10)
        elif (v, u) in P:
            edge_color.append('g')
            edge_width.append(10)
        else:
            edge_color.append('k')
            edge_width.append(1)
            
    for (u, v) in G.edges():
        assert (u, v) in Ninv

def fig1_network():
    """ Manually builds the Figure 1 networks. """
    G = nx.grid_2d_graph(11,11)
    
    G.graph['name'] = 'fig1'
    G.graph['nests'] = [(3,2), (8, 3)]

    for i,u in enumerate(G.nodes_iter()):
        M[i] = u
        Minv[u] = i

    for u,v in G.edges_iter(): G[u][v]['weight'] = MIN_PHEROMONE

    # horizontal edges.
    G.remove_edge((1,9),(2,9))
    G.remove_edge((3,9),(4,9))
    G.remove_edge((6,9),(7,9))
    G.remove_edge((8,9),(9,9))
    G.remove_edge((1,8),(2,8))
    G.remove_edge((8,8),(9,8))
    G.remove_edge((5,7),(6,7))
    G.remove_edge((9,7),(10,7))
    G.remove_edge((1,6),(2,6))
    G.remove_edge((5,6),(6,6))
    G.remove_edge((9,6),(10,6))
    G.remove_edge((1,5),(2,5))
    G.remove_edge((5,5),(6,5))
    G.remove_edge((7,5),(8,5))
    G.remove_edge((0,4),(1,4))
    G.remove_edge((2,4),(3,4))
    G.remove_edge((5,4),(6,4))
    G.remove_edge((7,4),(8,4))
    G.remove_edge((0,3),(1,3))
    #G.remove_edge((2,3),(3,3))
    G.remove_edge((3,3),(4,3)) # e
    G.remove_edge((3,2),(4,2))
    G.remove_edge((6,2),(7,2))
    G.remove_edge((0,1),(1,1))
    G.remove_edge((3,1),(4,1))
    G.remove_edge((8,1),(9,1))

    # vertical edges.
    G.remove_edge((1,0),(1,1))
    G.remove_edge((1,3),(1,4))
    G.remove_edge((1,5),(1,6))
    G.remove_edge((1,9),(1,8))
    #G.remove_edge((2,3),(2,4))
    G.remove_edge((2,5),(2,6))
    G.remove_edge((2,8),(2,9))
    G.remove_edge((3,1),(3,2))
    #G.remove_edge((3,3),(3,4))
    G.remove_edge((3,9),(3,10))
    G.remove_edge((4,1),(4,2))
    G.remove_edge((4,9),(4,10))
    G.remove_edge((5,4),(5,5))
    G.remove_edge((5,6),(5,7))
    G.remove_edge((6,2),(6,3))
    G.remove_edge((6,4),(6,5))
    G.remove_edge((6,6),(6,7))
    G.remove_edge((6,9),(6,10))
    G.remove_edge((7,2),(7,3))
    G.remove_edge((7,4),(7,5))
    G.remove_edge((7,9),(7,10))
    G.remove_edge((8,0),(8,1))
    G.remove_edge((8,4),(8,5))
    G.remove_edge((8,8),(8,9))
    G.remove_edge((9,0),(9,1))
    G.remove_edge((9,6),(9,7))
    G.remove_edge((9,8),(9,9))
    G.remove_edge((4,2),(4,3))
    G.remove_edge((4,3),(4,4))
    
    P.append(((7, 3), (8, 3)))
    P.append(((6, 3), (7, 3)))
    P.append(((5, 3), (6, 3)))
    P.append(((4, 3), (5, 3)))
    P.append(((3, 2), (3, 3)))

    init_graph(G)
    
    return G

def simple_network():
    '''
    Manually builds a simple network with 3 disjoint paths between nest and target
    '''
    G = nx.grid_2d_graph(8, 6)
    
    G.graph['name'] = 'simple'
    G.graph['nests'] = [(0, 3), (7, 3)]
        
    for j in xrange(6):
        if j < 5:
            G.remove_edge((0, j), (0, j + 1))
            G.remove_edge((7, j), (7, j + 1))
        
        if j != 3:
            G.remove_edge((0, j), (1, j))
            G.remove_edge((6, j), (7, j))
    
    
    for j in [1, 2, 4]:
        for k in xrange(1, 6):
            G.remove_edge((k, j), (k + 1, j))
            if 2 <= k <= 5:
                try:
                    G.remove_edge((k, j), (k, j + 1))
                except:
                    pass
                try:
                    G.remove_edge((k, j), (k, j - 1))
                except:
                    pass
    
    for j in xrange(7):
        if j != 5:
            P.append(((j, 3), (j + 1, 3)))
    
    G.remove_edge((5, 3), (6, 3))
                                            
    init_graph(G)
        
    return G

def full_grid():
    '''
    Manually builds a full 11x11 grid graph, puts two nests at opposite ends of the middle
    of the grid, and removes the very middle edge
    '''
    G = nx.grid_2d_graph(11,11)
    
    G.graph['name'] = 'full'
    G.graph['nests'] = [(0, 5), (10, 5)]
    
    G.remove_edge((4, 5), (5, 5))
    
    for i in range(10):
        if i != 4:
            P.append(((i, 5), (i + 1, 5)))

    init_graph(G)
    
    return G

def er_network(p=0.5):
    G = nx.grid_2d_graph(11, 11)
    
    G.graph['name'] = 'er'
    G.graph['nests'] = [(0, 5), (10, 5)]
    
    for u in G.nodes():
        for v in G.nodes():
            if u == nest and v == target:
                continue
            if v == nest and u == target:
                continue
            if u != v:
                if random() <= p:
                    G.add_edge(u, v)
                else:
                    if G.has_edge(u, v):
                        G.remove_edge(u, v)
    if not nx.has_path(G, nest, target):
        return None
    short_path = nx.shortest_path(G, nest, target)
    if len(short_path) <= 3:
        return None
    #print short_path
    idx = choice(range(1, len(short_path) - 1))
    #print idx
    G.remove_edge(short_path[idx], short_path[idx + 1])
    for i in xrange(idx):
        P.append((short_path[i], short_path[i + 1]))
    for i in xrange(idx + 1, len(short_path) - 1):
        P.append((short_path[i], short_path[i + 1]))
    #print P
        
    if not nx.has_path(G, nest, target):
        return None
    
    for i,u in enumerate(G.nodes_iter()):
        M[i] = u
        Minv[u] = i
        pos[u] = [u[0],u[1]] # position is the same as the label.

        if (u[0] == nest) or (u == target):
            node_size.append(100)
            node_color.append('r')
        else:
            node_size.append(10)
            node_color.append('k') 
        
    for u,v in G.edges_iter():
        G[u][v]['weight'] = MIN_PHEROMONE
        if (u, v) in P or (v, u) in P:
            edge_color.append('g')
            edge_width.append(10)
        else:
            edge_color.append('k')
            edge_width.append(1)
    
    for i, (u,v) in enumerate(G.edges()):
        Ninv[(u, v)] = i
        N[i] = (u, v)        
        Ninv[(v, u)] = i
            
    return G

def color_path(G, path, c, w, figname):
    """
    Given a path, colors that path on the graph and then outputs the colored path to a
    file
    """
    colors, widths = edge_color[:], edge_width[:]
    for i in xrange(len(path) - 1):
        edge = (path[i], path[i + 1])
        index = None
        try:
            index = Ninv[edge]
        except KeyError:
            index = Ninv[(path[i + 1], path[i])]
        colors[index] = c
        widths[index] += w
        
    nx.draw(G, pos=pos, with_labels=False, node_size=node_size, edge_color=colors, node_color=node_color, width=widths)
    PP.draw()
    #PP.show()
    PP.savefig(figname)
    PP.close()
    
def color_graph(G, c, w, figname):
    '''
    Draws the current graph and colors all the edges with pheromone, to display the
    pheromone network the ants have constructed at some point in time
    
    G - the networkx Graph object to be drawn
    
    c - the color to use for pheromone edges
    
    w - the scaling factor for edge weights.  If the edge widths are set directly equal to
        the edge weights, the edge widths will become prohibitively big and ruin the picture
        this scaling factor allows the edge widths to be proportional to the edge weights
        while capping the size of the largest edge.  Thus, this value should be a constant
        factor times the weight of the highest edge in the graph at the time of drawing.
        All edge weights and resulting widths are normalized by this factor.
        
    figname - the name to which to save the figure
    '''
    colors, widths = edge_color[:], edge_width[:]
    #unique_weights = set()
    for u, v in G.edges():
        index = None
        try:
            index = Ninv[(u, v)]
        except KeyError:
            index = Ninv[(v, u)]
        colors[index] = c
        wt = G[u][v]['weight']
        width = wt * w
        widths[index] = width
        #if width > 0:
            #print u, v, width
        #unique_weights.add(wt)
    #print len(unique_weights)
    nx.draw(G, pos=pos, with_labels=False, node_size=node_size, edge_color=colors, node_color=node_color, width=widths)
    PP.draw()
    #PP.show()
    PP.savefig(figname + '.png', format='png')
    PP.close()

def edge_weight(G, u, v):
    return sum(G[u][v]['units'])

def check_graph(G):
    for u, v in G.edges_iter():
        weight = G[u][v]['weight']
        assert weight >= MIN_PHEROMONE
        wt = 0
        for unit in G[u][v]['units']:
            assert unit > MIN_PHEROMONE
            wt += unit
        assert wt == weight

def decay_units(G, u, v, decay):
    nonzero_units = []
    for unit in G[u][v]['units']:
        unit = max(unit - decay, MIN_PHEROMONE)
        assert unit >= MIN_PHEROMONE
        if unit > MIN_PHEROMONE:
            nonzero_units.append(unit)
    G[u][v]['units'] = nonzero_units

def decay_edges(G, nonzero_edges, decay):
    zero_edges = []
    for i in nonzero_edges:
        u, v = N[i]
        decay_units(G, u, v, decay)
        wt = edge_weight(G, u, v)
        assert wt >= MIN_PHEROMONE
        G[u][v]['weight'] = wt
        if wt == MIN_PHEROMONE:
            zero_edges.append(i)
    return zero_edges
    
def decay_graph(G, decay):
    for u, v in G.edges_iter():
        decay_units(G, u, v, decay)
        wt = edge_weight(G, u, v)
        assert wt >= MIN_PHEROMONE
        G[u][v]['weight'] = wt
        '''
        wt = G[u][v]['weight']
        assert wt >= MIN_PHEROMONE
        x = max(MIN_PHEROMONE, wt - (decay * seconds))
        assert wt >= MIN_PHEROMONE
        G[u][v]['weight'] = x
        '''

def get_weights(G, start, candidates):
    '''
    Returns an array containing all the edge weights in the graph
    '''
    weights = map(lambda x : G[start][x]['weight'], candidates)
    return array(weights)
    
def rand_edge(G, start, candidates = None):
    '''
    Pick an ant's next edge.  Given the current vertex and possibly the list of candidates
    picks the next edge based on the pheromone levels.  In particular, if S is the sum of
    the total weights of all edges adjacent to start, then the function picks edge
    (start, u) with probability w(start, u) / S
    '''
    if candidates == None: 
        assert start != None
        candidates = G.neighbors(start)
    weights = get_weights(G, start, candidates)
    weights = weights / float(sum(weights))
    next = candidates[choice(len(candidates),1,p=weights)[0]]
    return next

def max_edge(G, start, candidates=None):
    '''
    Picks the next edge according to the max edge model.  Finds all adjacent edges that 
    are of maximal weight (among the set of neighboring edges).  Picks uniformly among all
    these maximal edges.
    '''
    if candidates == None:
        assert start != None
        candidates = G.neighbors(start)
    weights = get_weights(G, start, candidates)
    assert len(weights) == len(candidates)
    max_weight = max(weights)
    max_neighbors = []
    for i in xrange(len(weights)):
        w = weights[i]
        if w == max_weight:
            max_neighbors.append(candidates[i])
    next = choice(len(max_neighbors))
    next = max_neighbors[next]
    return next

def pheromone_subgraph(G, origin=None, destination=None):
    '''
    
    '''
    G2 = nx.Graph()
    for u, v in G.edges_iter():
        if G[u][v]['weight'] > MIN_PHEROMONE:
            G2.add_edge(u, v)
    if origin not in G2:
        G2.add_node(origin)
    if destination not in G2:
        G2.add_node(destination)
    return G2
    
def pheromone_paths(G, origin, destination, limit=15):
    G2 = pheromone_subgraph(G, origin, destination)
    return list(nx.all_simple_paths(G2, origin, destination, limit))

def pheromone_connectivity(G, origin, destination, limit=15):
    G2 = pheromone_subgraph(G, origin, destination)
    return len(list(nx.all_simple_paths(G2, origin, destination, limit)))

def has_pheromone_path(G, origin, destination):
    G2 = pheromone_subgraph(G, origin, destination)
    return nx.has_path(G2, origin, destination)
    
def next_edge(G, start, explore_prob, strategy='uniform', prev=None, dest=None, \
              search==True, backtrack=False):
    unexplored = []
    explored = []
    neighbors = G.neighbors(start)
    if (dest != None) and (dest in neighbors):
        return dest, False
    max_wt = float("-inf")
    for neighbor in neighbors:
        wt = G[start][neighbor]['weight']
        max_wt = max(wt, max_wt)
        if wt == MIN_PHEROMONE:
            unexplored.append(neighbor)
        else:
            explored.append(neighbor)
    
    candidates = explored + unexplored
    if candidates == [prev]:
        return prev, False 
        
    if (not backtrack) and (prev != None) and (len(explored) > 1):
        assert prev in explored
        explored.remove(prev)
        
    if prev != None:
        assert prev in candidates
        if prev in explored:
            explored.remove(prev)
        else:
            unexplored.remove(prev)
    
    if explore_prob == 0 and len(explored) == 0:
        return prev, False
        
    flip = random()
    max_mode = (strategy == 'max') or ((strategy == 'hybrid') and (search == False))
    if (flip < explore_prob and len(unexplored) > 0) or (len(explored) == 0):
        if max_mode:
            for e in explored:
                if G[start][e]['weight'] < max_wt:
                    unexplored.append(e)
        next = choice(len(unexplored))
        next = unexplored[next]
        return next, True
    
    assert len(explored) > 0
    if max_mode:
        return max_edge(G, start, explored), False
    else:
        return rand_edge(G, start, explored), False

def count_nonzero(G, curr):
    count = 0
    for neighbor in G.neighbors(curr):
        if G[curr][neighbor]['weight'] > MIN_PHEROMONE:
            count += 1
    return count
    
def path_weight(G, path):
    path = list(path)
    weight = 0
    for i in range(len(path) - 1):
        source = path[i]
        dest = path[i + 1]
        wt = G[source][dest]['weight']
        assert wt > MIN_PHEROMONE
        weight += wt
    return weight
    
def path_mean_weight(G, path):
    path = list(path)
    weight = 0.0
    for i in range(len(path) - 1):
        source = path[i]
        dest = path[i + 1]
        wt = G[source][dest]['weight']
        assert wt > MIN_PHEROMONE
        weight += wt
    return weight / len(path)
    
def path_score(G, path):
    weight = path_weight(G, path)
    length = len(path)
    return weight / float(length)

def mean_path_score(G, paths):
    paths = list(paths)
    if len(paths) == 0:
        return 0
    scores = map(lambda path : path_score(G, path), paths)
    return PP.mean(scores)
    
def all_paths_score(G, origin, destination, limit=15):
    subgraph = pheromone_subgraph(G, origin, destination)
    paths = nx.all_simple_paths(G, nest, target, limit)
    return mean_path_score(G, paths)
    
def pheromone_cost(G):
    G2 = nx.Graph()
    for u, v in G.edges_iter():
        if G[u][v]['weight'] > MIN_PHEROMONE:
            G2.add_edge(u, v)
    return G2.number_of_edges()
    
def vertex_entropy(G, vertex, explore_prob, prev=None):
    assert 0 < explore_prob < 1
    nonzero = []
    zero = []
    for n in G.neighbors(vertex):
        if n != prev:
            w = G[vertex][n]['weight']
            if w == 0:
                zero.append(explore_prob)
            else:
                nonzero.append(w)
    total = float(sum(nonzero))
    for i in xrange(len(nonzero)):
        nonzero[i] /= total
        nonzero[i] *= (1 - explore_prob)
    for i in xrange(len(zero)):
        zero[i] /= len(zero)
    probs = zero + nonzero
    return entropy(probs)
    
def choice_prob(G, source, dest, explore_prob, prev=None):
    neighbors = G.neighbors(source)
    assert dest in neighbors
    assert G[source][dest]['weight'] > 0
    total = 0.0
    for n in neighbors:
        if n != prev:
            total += G[source][n]['weight']
    return (1 - explore_prob) * (G[source][dest]['weight'] / total)
    
def path_prob(G, path, explore_prob):
    prob = 1
    prev = None
    for i in xrange(len(path) - 1):
        source = path[i]
        dest = path[i + 1]
        prob *= choice_prob(G, source, dest, explore_prob, prev)
        prev = source
    return prob
    
def path_prob_no_explore(G, path):
    return path_prob(G, path, explore_prob=0)
    
def path_entropy(G, path, explore_prob):
    probs = []
    prev = None
    for i in xrange(len(path) - 1):
        source = path[i]
        dest = path[i + 1]
        probs.append(choice_prob(G, source, dest, explore_prob, prev))
        prev = source
    return entropy(probs)

def pruning_plot(costs, figname, max_cost=None):
    if max_cost == None:
        max_cost = max(costs)
    assert max_cost in costs
    costs = PP.array(costs)
    costs /= float(max_cost)
    assert 1 in costs
    PP.plot(range(len(costs)), costs)
    PP.xlabel('time steps')
    PP.ylabel('proportion of edges in use')
    PP.savefig(figname + '.png', format='png')

def repair(G, strategy, pheromone_add, pheromone_decay, explore_prob, \
            num_ants=100, max_steps=10000, num_iters=0, print_graph=False, video=False, \
            nframes=200, video2=False, cost_plot=False, pruning_plot=False, \
            path_pruning_plot=False):
    """ """
    
    graph_name = G.graph['name']
    nest1 = G.graph['nest1']
    nest2 = G.graph['nest2']
    
    
    def next_destination(origin):
        if origin == nest1:
            return nest2
        return nest1
    
    num_edges = G.size()
    
    nframes = min(nframes, max_steps)
    
    pher_str = "%d, %f, %f, " % (num_ants, explore_prob, pheromone_decay)

    for iter in xrange(num_iters):
        if video:
            fig = PP.figure()
            
        nonzero_edges = set()
        for u, v in G.edges_iter():
            G[u][v]['weight'] = MIN_PHEROMONE
            G[u][v]['units'] = []
        for u, v in P:
            G[u][v]['weight'] += pheromone_add * INIT_WEIGHT_FACTOR
            G[u][v]['units'] += [pheromone_add] * INIT_WEIGHT_FACTOR
            nonzero_edges.add(Ninv[(u, v)])
        
        if iter == 0 and print_graph:
            color_graph(G, 'g', pheromone_thickness, "graph_before")
        print str(iter) + ": " + pher_str
        explore = defaultdict(bool)
        paths = {}
        walks = {}
        destinations = {}
        origins = {}
        edge_weights = defaultdict(list)
        deadend = {}
        
        connect_time = -1
        before_paths = after_paths = 0
                
        for ant in xrange(num_ants):
            if ant % 2 == 0:
                origins[ant] = nest
            else:
                origins[ant] = target
            origin = origins[ant]
            destinations[ant] = next_destination(origin)
            paths[ant] = [origin, origin]
            walks[ant] = [origin, origin]
            
        steps = 1
        max_weight = MIN_PHEROMONE
        unique_weights = set()
        max_cost = 0
        costs = []
        
        while steps <= max_steps:
            #check_graph(G)
            cost = pheromone_cost(G)
            max_cost = max(max_cost, cost)
            costs.append(cost)
            G2 = G.copy()
                
            for u, v in G.edges():
                index = None
                try:
                    index = Ninv[(u, v)]
                except KeyError:
                    index = Ninv[(v, u)]
                wt = G[u][v]['weight']
                unique_weights.add(wt)
                max_weight = max(max_weight, wt)
                if wt == MIN_PHEROMONE:
                    edge_weights[index].append(None)
                else:
                    edge_weights[index].append(wt)
            for j in xrange(num_ants):
                curr = paths[j][-1]
                prev = paths[j][-2]
                next = None
                
                n = G.neighbors(curr)
                if curr != prev:
                    n.remove(prev)
                if len(n) == 0:
                    deadend[j] = (curr not in [nest, target])
                    #print curr, deadend[j]
                elif len(n) > 1:
                    deadend[j] = False
                if j / 2 < steps:
                    if (prev == curr) or (curr == origins[j]):
                        prev = None
                    next, ex = next_edge(G, curr, explore_prob=explore_prob, prev=prev, dest=destinations[j])
                    add_amt = pheromone_add
                    add_neighbor = next
                    if ex:
                        add_amt *= 2
                        next = curr
                    if not deadend[j]:
                        G2[curr][add_neighbor]['weight'] += add_amt
                        G2[curr][add_neighbor]['units'].append(add_amt)
                        nonzero_edges.add(Ninv[(curr, add_neighbor)])
                else:
                    next = curr
                paths[j].append(next)
                if next == destinations[j]:
                    origins[j], destinations[j] = destinations[j], origins[j]
                    
                '''
                if explore[j]:
                    paths[j].append(prev)
                    explore[j] = False
                    if not deadend[j]:
                        G2[curr][prev]['weight'] += pheromone_add
                        G2[curr][prev]['units'].append(pheromone_add)
                        nonzero_edges.add(Ninv[(curr, prev)])
                else:
                    if curr == origins[j]:
                        prev = None
                    next, ex = next_edge(G, curr, explore_prob=explore_prob, prev=prev)
                    explore[j] = ex
                    paths[j].append(next)
                    if not deadend[j]:
                        G2[curr][next]['weight'] += pheromone_add
                        G2[curr][next]['units'].append(pheromone_add)
                        nonzero_edges.add(Ninv[(curr, next)])
                    if next == destinations[j]:
                        origins[j], destinations[j] = destinations[j], origins[j]
                '''
                                    
            #decay_graph(G2, pheromone_decay)
            zero_edges = decay_edges(G2, nonzero_edges, pheromone_decay)
            nonzero_edges.difference_update(zero_edges)
                
            G = G2
            steps += 1
                    
        if print_graph:        
            color_graph(G, 'g', (pheromone_add / max_weight), "graph_after_simple_deadend%d_e%0.2fd%0.2f" % (max_steps, explore_prob, pheromone_decay))
            print "graph colored"
        
        e_colors = edge_color[:]
        e_widths = edge_width[:]
        n_colors = node_color[:]
        n_sizes = node_size[:]
        
        n_colors[Minv[target]] = 'm'
        n_colors[Minv[nest]] = 'y'
        
        n_sizes[Minv[target]] = n_sizes[Minv[nest]] = 100
        
                
        def init():
            nx.draw(G, pos=pos, with_labels=False, node_size=n_sizes, edge_color=e_colors, node_color=n_colors, width=e_widths)
        
        def redraw(frame):
            PP.clf()
            frame = min(frame, max_steps)
            print frame
            e_colors = ['k'] * len(edge_color)
            e_widths = [1] * len(edge_width)
            n_colors = ['r'] * len(node_color)
            n_sizes = [10] * len(node_size)
            
            ax = PP.gca()
            
            for n in xrange(num_ants):             
                node = paths[n][frame]
                index = Minv[node]
                n_colors[index] = 'k'
                n_sizes[index] += ant_thickness
                            
            if frame > 0:
                frame -= 1
                max_units = max_weight / pheromone_add
                for index in edge_weights:
                    wt = edge_weights[index][frame]
                    if wt != None:
                        units = edge_weights[index][frame]
                        e_widths[index] = 1 + 5 * (units / max_units)
                        e_colors[index] = 'g'
                                        
            n_colors[Minv[target]] = 'm'
            n_colors[Minv[nest]] = 'y'
            n_sizes[Minv[target]] = max(n_sizes[Minv[target]], 100)
            n_sizes[Minv[nest]] = max(n_sizes[Minv[nest]], 100)

            nx.draw(G, pos=pos, with_labels=False, node_size=n_sizes, edge_color=e_colors, \
                    node_color=n_colors, width=e_widths)
            f = PP.draw()
            return f,
        
        if nframes == -1:
            nframes = steps
        
        if video:    
            ani = animation.FuncAnimation(fig, redraw, init_func=init, frames=nframes, \
                                          interval = 1000)
            ani.save("ant_deviate_simple_deadend" + str(iter) + ".mp4")
            
        cost = pheromone_cost(G)
        costs.append(cost)
        max_cost = max(max_cost, cost)
        costs = PP.array(costs)
        if cost_plot:
            figname = "pruning/pruning_simple_deadend%d_e%0.2fd%0.2f" % (max_steps, explore_prob, pheromone_decay)
            pruning_plot(costs, figname, max_cost)
            return None

        if not video2:
            continue

        # Output results.
        has_path = has_pheromone_path(G, nest, target)
            
        journey_times = []
        journey_lengths = []
        walk_counts = defaultdict(int)
        total_steps = 0
        print "new ants"
        successful_walks = 0
        failed_walks = 0
        for new_ant in xrange(10000):
            curr = nest
            prev = None
            ex = False
            steps = 0
            walk = []
            if not has_path:
                #data_file2.write('%f, %f, %d\n' % (explore_prob, pheromone_decay, -1))
                failed_walks += 1
                continue
            assert has_path
            while curr != target and steps <= 1000:
                steps += 1
                total_steps += 1
                next = None
                walk.append(curr)
                next, ex = next_edge(G, curr, explore_prob=0, prev=prev)

                prev = curr
                curr = next
            if curr != target:
                steps = -1
                failed_walks += 1
            else:
                journey_times.append(steps)
                walk_counts[tuple(walk)] += 1
                successful_walks += 1
        
        
        def init2():
            PP.clf()
            e_colors = ['k'] * len(edge_color)
            e_widths = [1] * len(edge_width)
            n_colors = ['r'] * len(node_color)
            n_sizes = [10] * len(node_size)
            
            ax = PP.gca()
            
            max_units = max_weight / pheromone_add
            for index in edge_weights:
                wt = edge_weights[index][-1]
                if wt != None:
                    units = edge_weights[index][-1] / pheromone_add
                    e_widths[index] = 1 + 5 * (units / max_units)
                    e_colors[index] = 'g'
            
            n_colors[Minv[target]] = 'm'
            n_colors[Minv[nest]] = 'y'
            n_sizes[Minv[target]] = max(n_sizes[Minv[target]], 100)
            n_sizes[Minv[nest]] = max(n_sizes[Minv[nest]], 100)
            
            nx.draw(G, pos=pos, with_labels=False, node_size=n_sizes, edge_color=e_colors, \
                    node_color=n_colors, width=e_widths)

        
        def redraw2(frame):
            PP.clf()
            e_colors = ['k'] * len(edge_color)
            e_widths = [1] * len(edge_width)
            n_colors = ['r'] * len(node_color)
            n_sizes = [10] * len(node_size)
            
            ax = PP.gca()
            print frame
            
            max_units = max_weight / pheromone_add
            for index in edge_weights:
                wt = edge_weights[index][-1]
                if wt != None:
                    units = edge_weights[index][-1] / pheromone_add
                    e_widths[index] = 1 + 5 * (units / max_units)
                    e_colors[index] = 'g'
            
            n_colors[Minv[target]] = 'm'
            n_colors[Minv[nest]] = 'y'
            n_sizes[Minv[target]] = max(n_sizes[Minv[target]], 100)
            n_sizes[Minv[nest]] = max(n_sizes[Minv[nest]], 100)
            
            curr_index = 0
            curr_walk = None
            for walk in walk_counts.keys():
                curr_walk = walk
                if curr_index + len(walk) > frame:
                    break
                curr_index += len(walk)
            
            curr_pos = curr_walk[frame - curr_index]        
            n_sizes[Minv[curr_pos]] = 100
            n_colors[Minv[curr_pos]] = 'b'
                

            nx.draw(G, pos=pos, with_labels=False, node_size=n_sizes, edge_color=e_colors, \
                    node_color=n_colors, width=e_widths)
            f = PP.draw()
            return f,
        
        if video2:    
            ani = animation.FuncAnimation(fig, redraw2, init_func=init2, frames=total_steps, \
                                          interval = 1000)
            ani.save("ant_deviate_simple_deadend" + str(iter) + "a.mp4")
            
        print iter + 1   
    

def main():
    start = time.time()
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(asctime)s -- %(message)s'
    )
    
    graph_choices = ['fig1', 'full', 'simple', 'simple_weighted', 'simple_multi']
    strategy_choices = ['uniform', 'max', 'hybrid']
    

    usage="usage: %prog [options]"
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("-g", "--graph", dest='graph', choices=graph_choices, default='full',\
                        help="graph to run algorithm on")
    parser.add_argument('-s', '--strategy', dest='strategy', choices=strategy_choices,\
                        default='uniform', help="strategy to run")
    parser.add_argument("-x", "--repeats", type=int, dest="iterations", default=1,\
                        help="number of iterations") 
    parser.add_argument("-a", "--add", type=float, dest="pheromone_add", default=1,\
                        help="amt of phermone added")
    parser.add_argument("-d", "--decay", action="store", type=float, dest="pheromone_decay", \
                        default=0.05, help="amt of pheromone decay")
    parser.add_argument("-n", "--number", action="store", type=int, dest="num_ants", \
                        default=100, help="number of ants")
    parser.add_argument("-pg", "--print_graph", action="store_true", dest="print_graph", \
                        default=False)
    parser.add_argument("-v", "--video", action="store_true", dest="video", default=False)
    parser.add_argument("-v2", "--video2", action="store_true", dest="video2", default=False)
    parser.add_argument("-f", "--frames", action="store", type=int, dest="frames", \
                        default=200)
    parser.add_argument("-e", "--explore", action="store", type=float, dest="explore", \
                        default=0.1)
    parser.add_argument("-m", "--max_steps", action="store", type=int, dest="max_steps", \
                        default=3000)
    parser.add_argument("-c", "--cost_plot", action="store_true", dest="cost_plot", default=False)
    parser.add_argument('-b', '--backtrack', action='store_true', dest='backtrack', default=False)

    args = parser.parse_args()
    # ===============================================================

    # ===============================================================
    graph = args.graph
    strategy = args.strategy
    num_iters = args.iterations
    pheromone_add = args.pheromone_add
    pheromone_decay = args.pheromone_decay
    num_ants = args.num_ants
    print_graph = args.print_graph
    video = args.video
    video2 = args.video2
    frames = args.frames
    explore = args.explore
    max_steps = args.max_steps
    cost_plot = args.cost_plot
    backtrack = args.backtrack

    # Build network.
    if graph == 'fig1':
        G = fig1_network()
    elif graph == 'simple':
        G = simple_network()
    elif graph == 'full':
        G = full_grid()

    '''
    nx.draw(G, pos=pos, with_labels=False, node_size=node_size, edge_color=edge_color, \
            node_color=node_color, width=edge_width)
    PP.draw()
    print "show"
    PP.show()
    #PP.savefig("%s.pdf" % G.graph['name'])
    #PP.close()
    '''

    # Run recovery algorithm.
    repair(G, strategy, pheromone_add, pheromone_decay, explore_prob, num_ants, max_steps, \
           num_iters, print_graph, video, nframes, video2, cost_plot, pruning_plot, \
           path_pruning_plot, backtrack)
    
    # =========================== Finish ============================
    logging.info("Time to run: %.3f (mins)" %((time.time()-start) / 60))


if __name__ == "__main__":
    main()