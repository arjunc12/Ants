#!/usr/bin/env python

from __future__ import division
import networkx as nx
import time,logging
from optparse import OptionParser
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

N = {}    # edge -> edge id
Ninv = {} # edge id -> edge

#MAX_STEPS= 10000
MIN_PHEROMONE = 0
pos = {}
node_color,node_size = [],[]
edge_color,edge_width = [],[]
P = []
path_thickness = 10
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

MAX = True
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



def fig1_network():
    """ Manually builds the Figure 1 networks. """
    G = nx.grid_2d_graph(11,11)

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

    # Draw the network.
    for u in G.nodes():
        pos[u] = [u[0],u[1]] # position is the same as the label.

        # nests
        # if u[0] == 5 and u[1] == 8:
#             node_size.append(100)
#             node_color.append('r')
        if u[0] == 3 and u[1] == 2:
            node_size.append(100)
            node_color.append('r')
        elif u[0] == 8 and u[1] == 3:
            node_size.append(100)
            node_color.append('r')
        else:
            node_size.append(10)
            node_color.append('k')

    for i, (u,v) in enumerate(G.edges()):
        Ninv[(u, v)] = i
        N[i] = (u, v)        
        Ninv[(v, u)] = i
        
        # if u == (5,8) and v == (6,8): edge_color.append('r')
#         elif u == (6,7) and v == (6,8): edge_color.append('r')
#         elif u == (6,7) and v == (7,7): edge_color.append('r')
#         elif u == (7,6) and v == (7,7): edge_color.append('r')
#         elif u == (7,6) and v == (8,6): edge_color.append('r')
#         elif u == (8,5) and v == (8,6): edge_color.append('r')
#         elif u == (8,5) and v == (9,5): edge_color.append('r')
#         elif u == (9,4) and v == (9,5): edge_color.append('r')
#         elif u == (9,4) and v == (8,4): edge_color.append('r')
#         elif u == (8,3) and v == (8,4): edge_color.append('r')
        if u == (7,3) and v == (8,3): 
            edge_color.append('r')
        elif u == (7,3) and v == (6,3): 
            edge_color.append('r')
        elif u == (5,3) and v == (6,3): 
            edge_color.append('r')
        elif u == (5,3) and v == (4,3): 
            edge_color.append('r')
        elif u == (3,2) and v == (3,3): 
            edge_color.append('r')
        else:
            edge_color.append('k')

        if edge_color[-1] == 'r': 
            edge_width.append(2)
            P.append((u,v))
        else:
            edge_width.append(1)
            
    for (u, v) in G.edges():
        assert (u, v) in Ninv
    return G

def simple_network():
    '''
    Manually builds a simple network with 3 disjoint paths between nest and target
    '''
    G = nx.grid_2d_graph(8, 6)
        
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
                                            
    for i,u in enumerate(G.nodes_iter()):
        M[i] = u
        Minv[u] = i
                
    # Draw the network.
    for u in G.nodes():
        pos[u] = [u[0],u[1]] # position is the same as the label.

        # nests
        # if u[0] == 5 and u[1] == 8:
#             node_size.append(100)
#             node_color.append('r')
        if u[0] == 0 and u[1] == 3:
            node_size.append(100)
            node_color.append('r')
        elif u[0] == 7 and u[1] == 3:
            node_size.append(100)
            node_color.append('r')
        else:
            node_size.append(10)
            node_color.append('k')
    
    for i, (u, v) in enumerate(G.edges()):
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
        
    return G

def full_grid():
    '''
    Manually builds a full 11x11 grid graph, puts two nests at opposite ends of the middle
    of the grid, and removes the very middle edge
    '''
    G = nx.grid_2d_graph(11,11)
    G.remove_edge((4, 5), (5, 5))

    for i,u in enumerate(G.nodes_iter()):
        M[i] = u
        Minv[u] = i

    for u,v in G.edges_iter(): G[u][v]['weight'] = MIN_PHEROMONE
    
    for u in G.nodes():
        pos[u] = [u[0],u[1]] # position is the same as the label.
        

        if u[0] == 0 and u[1] == 5:
            node_size.append(100)
            node_color.append('r')
        elif u[0] == 10 and u[1] == 5:
            node_size.append(100)
            node_color.append('r')
        else:
            node_size.append(10)
            node_color.append('k')

    for i, (u,v) in enumerate(G.edges()):
        Ninv[(u, v)] = i
        N[i] = (u, v)        
        Ninv[(v, u)] = i
        
        if u[1] == 5 and v[1] == 5:
            P.append((u, v))
            edge_color.append('g')
            edge_width.append(10)
        else:
            edge_color.append('k')

            edge_width.append(1)
            
    for (u, v) in G.edges():
        assert (u, v) in Ninv
    return G

def er_network(p=0.5):
    G = nx.grid_2d_graph(11, 11)
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
        widths[index] = width * pheromone_thickness
        #if width > 0:
            #print u, v, width
        #unique_weights.add(wt)
    #print len(unique_weights)
    nx.draw(G, pos=pos, with_labels=False, node_size=node_size, edge_color=colors, node_color=node_color, width=widths)
    PP.draw()
    #PP.show()
    PP.savefig(figname + '.png', format='png')
    PP.close()

def check_graph_weights(G):
    '''
    Ensure that no edges have weight lower than the minimum allowable weight
    '''
    for u, v in G.edges_iter():
        wt = G[u][v]['weight']
        assert wt >= MIN_PHEROMONE

def check_graph(G):
    for u, v in G.edges_iter():
        weight = G[u][v]['weight']
        assert weight >= MIN_PHEROMONE
        wt = 0
        for unit in G[u][v]['units']:
            assert unit > MIN_PHEROMONE
            wt += unit
        assert wt == weight

def decay_edges(G, nonzero_edges, decay):
    zero_edges = []
    for i in nonzero_edges:
        u, v = N[i]
        wt = G[u][v]['weight']
        assert wt > MIN_PHEROMONE
        x = max(MIN_PHEROMONE, wt - decay)
        assert x >= MIN_PHEROMONE
        G[u][v]['weight'] = x
        if x == MIN_PHEROMONE:
            zero_edges.append(Ninv[(u, v)])
    return zero_edges

def edge_weight(G, u, v):
    return sum(G[u][v]['units'])

def decay_units(G, u, v, decay):
    nonzero_units = []
    for unit in G[u][v]['units']:
        unit = max(unit - decay, MIN_PHEROMONE)
        assert unit >= MIN_PHEROMONE
        if unit > MIN_PHEROMONE:
            nonzero_units.append(unit)
    G[u][v]['units'] = nonzero_units
    
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
    
def next_edge(G, start, explore_prob=0.1, prev=None):
    unexplored = []
    explored = []
    neighbors = G.neighbors(start)
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
    '''
    if (not BACKTRACK) and (prev != None) and (len(explored) > 1):
        assert prev in explored
        explored.remove(prev)
    '''
    if prev != None:
        assert prev in candidates
        if prev in explored:
            explored.remove(prev)
        else:
            unexplored.remove(prev)
    
    if explore_prob == 0 and len(explored) == 0:
        return prev, False
        
    flip = random()
    if (flip < explore_prob and len(unexplored) > 0) or (len(explored) == 0):
        if MAX:
            for e in explored:
                if G[start][e]['weight'] < max_wt:
                    unexplored.append(e)
        next = choice(len(unexplored))
        next = unexplored[next]
        return next, True
    
    assert len(explored) > 0
    if MAX:
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
    
def walk_to_path(walk):
    assert len(walk) >= 2
    assert walk[0] != walk[-1]
    path = []
    visited = {}
    for i, node in enumerate(walk):
        if node not in visited:
            path.append(node)
            visited[node] = len(path) - 1
        else:
            prev = visited[node]
            for j in xrange(prev + 1, len(path)):
                del visited[path[j]]
            path = path[:prev + 1]
    assert len(path) >= 2
    assert path[0] == walk[0]
    assert path[-1] == walk[-1]
    
    return path

def deviate(G, num_iters, num_ants, pheromone_add, pheromone_decay, explore_prob, \
            print_graph=False, max_steps=3000, cost_plot=False):
    """ """
    # os.system("rm -f graph*.png")
    # Put ants at the node adjacent to e, at node (4,3).
    #bkpt = (4,3)
    #init = (5,3)
    #target = (0,5)
    #nest = (10,5)
    nest = (0, 3)
    target = (7, 3)
    
    def next_destination(prev):
        if prev == target:
            return nest
        return target
    
    num_edges = G.size()
    
    data_file = open('ant_deviate_max_simple_deadend%d.csv' % max_steps, 'a')
    pruning_file = open('ant_deviate_max_simple_deadend_pruning.csv', 'a')
    #path_pruning_file = open('ant_deviate_simple_deadend_path_pruning.csv', 'a')
    pher_str = "%d, %f, %f, " % (num_ants, explore_prob, pheromone_decay)
    # Repeat 'num_iters' times 
    for iter in xrange(num_iters):
        for u, v in G.edges_iter():
            G[u][v]['weight'] = MIN_PHEROMONE
            G[u][v]['units'] = []
        for u, v in P:
            G[u][v]['weight'] += pheromone_add * INIT_WEIGHT_FACTOR
            G[u][v]['units'] += [pheromone_add] * INIT_WEIGHT_FACTOR
        
        if iter == 0 and print_graph:
            color_graph(G, 'g', pheromone_thickness, "graph_before")
        print str(iter) + ": " + pher_str
        explore = defaultdict(bool)
        prevs = {}
        currs = {}
        destinations = {}
        origins = {}
        deadend = {}
        
        walks = {}
        path_counts = defaultdict(int)
        chosen_walk_counts = defaultdict(int)
        max_entropy = None
        max_walk_entropy = None
                
        connect_time = -1
        before_paths = after_paths = 0
                
        for ant in xrange(num_ants):
            if ant % 2 == 0:
                prevs[ant] = nest
                currs[ant] = (1, 3)
                destinations[ant] = target
                origins[ant] = nest
            else:
                prevs[ant] = target
                currs[ant] = (6, 3)
                destinations[ant] = nest
                origins[ant] = target 
            walks[ant] = [prevs[ant], currs[ant]]
            deadend[ant] = False
            
        steps = 1
        max_cost = 0
        costs = []
        curr_entropy = None
        curr_walk_entropy = None
        max_weight = MIN_PHEROMONE
        while steps <= max_steps:
            cost = pheromone_cost(G)
            max_cost = max(max_cost, cost)
            costs.append(cost)
            prun_write_items = [steps, cost]
            
            if curr_entropy != None:
                prun_write_items.append(curr_entropy)
            else:
                prun_write_items.append('')
                
            if curr_walk_entropy != None:
                prun_write_items.append(curr_walk_entropy)
            else:
                prun_write_items.append('')
                
            prun_str = ', '.join(map(str, prun_write_items))
            pruning_file.write(pher_str + prun_str + '\n')
            
            for u, v in G.edges():
                wt = G[u][v]['weight']
                max_weight = max(max_weight, wt)
            
            G2 = G.copy()
            for j in xrange(num_ants):
                curr = currs[j]
                prev = prevs[j]
                
                n = G.neighbors(curr)
                if curr != prev:
                    n.remove(prev)
                if len(n) == 0:
                    deadend[j] = (curr not in [nest, target])
                    #print curr, deadend[j]
                elif len(n) > 1:
                    deadend[j] = False
                
                if (prev == curr) or (curr == origins[j]):
                    prev = None
                next, ex = next_edge(G, curr, explore_prob=explore_prob, prev=prev)
                add_amt = pheromone_add
                add_neighbor = next
                if ex:
                    add_amt *= 2
                    next = curr
                if not deadend[j]:
                    G2[curr][add_neighbor]['weight'] += add_amt
                    G2[curr][add_neighbor]['units'].append(add_amt)
                walks[j].append(next)
                prevs[j] = curr
                currs[j] = next
                
                if next == destinations[j]:
                    origins[j], destinations[j] = destinations[j], origins[j]
                    
                    walk = walks[j]
                    chosen_walk_counts[tuple(walk)] += 1
                    
                    path = walk_to_path(walk)
                    if path[-1] == nest:
                        path = path[::-1]
                        assert path[-1] == target
                    path_counts[tuple(path)] += 1
                    
                    curr_entropy = entropy(path_counts.values())
                    curr_walk_entropy = entropy(chosen_walk_counts.values())
                    
                    if max_entropy == None:
                        max_entropy = curr_entropy
                    else:
                        max_entropy = max(max_entropy, curr_entropy)
                        
                    if max_walk_entropy == None:
                        max_walk_entropy = curr_walk_entropy
                    else:
                        max_walk_entropy = max(max_walk_entropy, curr_walk_entropy)    
                    
                    walks[j] = [origins[j]]
                
                '''
                if explore[j]:
                    prevs[j] = curr
                    currs[j] = prev
                    explore[j] = False
                    if not deadend[j]:
                        G2[curr][prev]['weight'] += pheromone_add
                        G2[curr][prev]['units'].append(pheromone_add)
                    walks[j].append(currs[j])
                else:
                    if curr == origins[j]:
                        prev = None
                    next, ex = next_edge(G, curr, explore_prob=explore_prob, prev=prev)
                    explore[j] = ex
                    prevs[j] = curr
                    currs[j] = next
                    if not deadend[j]:
                        G2[curr][next]['weight'] += pheromone_add
                        G2[curr][next]['units'].append(pheromone_add)
                    walks[j].append(next)
                    if next == destinations[j]:
                        origins[j], destinations[j] = destinations[j], origins[j]
                        
                        walk = walks[j]
                        chosen_walk_counts[tuple(walk)] += 1
                        
                        path = walk_to_path(walk)
                        if path[-1] == nest:
                            path = path[::-1]
                            assert path[-1] == target
                        path_counts[tuple(path)] += 1
                        
                        curr_entropy = entropy(path_counts.values())
                        curr_walk_entropy = entropy(chosen_walk_counts.values())
                        
                        if max_entropy == None:
                            max_entropy = curr_entropy
                        else:
                            max_entropy = max(max_entropy, curr_entropy)
                            
                        if max_walk_entropy == None:
                            max_walk_entropy = curr_walk_entropy
                        else:
                            max_walk_entropy = max(max_walk_entropy, curr_walk_entropy)    
                        
                        walks[j] = [origins[j]]
                '''
                                    
            decay_graph(G2, pheromone_decay)
                
            G = G2
            steps += 1
            if connect_time == -1 and has_pheromone_path(G, nest, target):
                connect_time = steps
        
        if print_graph:        
            color_graph(G, 'g', pheromone_add / max_weight, "graph_after_deviate_max_simple_deadend" + str(iter))
            
        cost = pheromone_cost(G)
        costs.append(cost)
        max_cost = max(max_cost, cost)
        costs = PP.array(costs)
        pruning = (max_cost - cost) / float(max_cost)
        if cost_plot:
            figname = "pruning/pruning_max_simple_deadend%d_e%0.2fd%0.2f" % (max_steps, explore_prob, pheromone_decay)
            pruning_plot(costs, figname, max_cost)
            return None
        
        path_pruning = None  
        if len(path_counts) > 0:
            curr_entropy = entropy(path_counts.values())
            max_entropy = max(max_entropy, curr_entropy)
            path_pruning = max_entropy - curr_entropy
            
        walk_pruning = None
        if len(chosen_walk_counts) > 0:
            curr_walk_entropy = entropy(chosen_walk_counts.values())
            max_walk_entropy = max(max_walk_entropy, curr_walk_entropy)
            walk_pruning = max_walk_entropy - curr_walk_entropy

        # Output results.
        path_lengths, revisits = [], []
        right_nest, wrong_nest = 0.0, 0.0
        hit_counts, miss_counts = [], []
        
        has_path = has_pheromone_path(G, nest, target)
        print "has path", has_path
        after_paths = pheromone_paths(G, nest, target, MAX_PATH_LENGTH)
        path_probs = []
        for path in after_paths:
            path_probs.append(path_prob_no_explore(G, path))
        
        min_etr = entropy(G.number_of_nodes() * [1.0 / G.number_of_nodes()])
        path_etr = min_etr
        if len(after_paths) > 0:
            path_etr = entropy(path_probs)
            
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
            
        walk_success_rate = float(successful_walks) / (successful_walks + failed_walks)
        mean_journey_time = mean(journey_times)
        med_journey_time = median(journey_times)
        
        
        walk_entropy = entropy(walk_counts.values())
        
        write_items = [int(has_path), cost]
        
        if len(path_probs) > 0 and path_etr != float("-inf"):
            write_items.append(path_etr)
        else:
            write_items.append('')
        
        if len(walk_counts.values()) > 0:
            write_items += [walk_entropy, mean_journey_time, med_journey_time]
        else:
            write_items += ['', '', '']
        
        write_items.append(walk_success_rate)
        
        write_items.append(pruning)
        
        if connect_time != -1:
            write_items.append(connect_time)
        else:
            write_items.append('')
        
        if path_pruning != None:
            write_items.append(path_pruning)
        else:
            write_items.append('')
            
        if curr_entropy != None:
            write_items.append(curr_entropy)
        else:
            write_items.append('')
         
        if walk_pruning != None:
            write_items.append(walk_pruning)
        else:
            write_items.append('')
            
        if curr_walk_entropy != None:
            write_items.append(curr_walk_entropy)
        else:
            write_items.append('')
        
        ant_str = ', '.join(map(str, write_items))
        line = pher_str + ant_str + '\n'
        data_file.write(line)
            
        print iter + 1          
    
    data_file.close()
    pruning_file.close()
    #path_pruning_file.close()
    

def main():
    start = time.time()
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(asctime)s -- %(message)s'
    )

    usage="usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-x", "--repeats", action="store", type="int", dest="iterations", default=10,help="number of iterations")
    parser.add_option("-a", "--add", action="store", type="float", dest="pheromone_add", default=MIN_ADD,help="amt of phermone added")
    parser.add_option("-d", "--decay", action="store", type="float", dest="pheromone_decay", default=0.0,help="amt of pheromone decay")
    parser.add_option("-n", "--number", action="store", type="int", dest="num_ants", default=10,help="number of ants")
    parser.add_option("-p", "--print_path", action="store_true", dest="print_path", default=False)
    parser.add_option("-g", "--print_graph", action="store_true", dest="print_graph", default=False)
    parser.add_option("-v", "--video", action="store_true", dest="video", default=False)
    parser.add_option("-f", "--frames", action="store", type="int", dest="frames", default=200)
    parser.add_option("-e", "--explore", action="store", type="float", dest="explore", default=0.1)
    parser.add_option("-m", "--max_steps", action="store", type="int", dest="max_steps", default=3000)
    parser.add_option("-c", "--cost_plot", action="store_true", dest="cost_plot", default=False)

    (options, args) = parser.parse_args()
    # ===============================================================

    # ===============================================================
    num_iters = options.iterations
    pheromone_add = options.pheromone_add
    pheromone_decay = options.pheromone_decay
    num_ants = options.num_ants
    print_path = options.print_path
    print_graph = options.print_graph
    video = options.video
    frames = options.frames
    explore = options.explore
    max_steps = options.max_steps
    cost_plot = options.cost_plot

    # Build network.
    #G = fig1_network()
    G = simple_network()
    #G = full_grid()

    #nx.draw(G,pos=pos,with_labels=False,node_size=node_size,edge_color=edge_color,node_color=node_color,width=edge_width)
    #PP.draw()
    #PP.show()
    #PP.savefig("fig_simple.pdf")
    #PP.close()

    # Run recovery algorithm.
    deviate(G,num_iters,num_ants,pheromone_add,pheromone_decay, explore, print_graph, \
            max_steps, cost_plot)

    
    # =========================== Finish ============================
    logging.info("Time to run: %.3f (mins)" %((time.time()-start) / 60))


if __name__ == "__main__":
    main()