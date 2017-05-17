#!/usr/bin/env python

from __future__ import division
import networkx as nx
import time,logging
from optparse import OptionParser
import argparse
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pylab as PP
from numpy.random import seed,choice, random
from numpy import mean,median, array, argmax, where, subtract
from collections import defaultdict
import os
from matplotlib import animation
from scipy.stats import spearmanr
from scipy.stats import entropy

from graphs import *
from choice_functions import *
from decay_functions import *

import sys
from random import randint

import os

SEED_DEBUG = False

SEED_MAX = 4294967295
SEED_VAL = randint(0, SEED_MAX)
if SEED_DEBUG:
    seed(SEED_VAL)
#seed(3305480832)

Minv = {} # node tuple -> node id
M = {}    # node id -> node tuple

Ninv = {}    # edge -> edge id
N = {}       # edge id -> edge

AFTER_GRAPH_THRESHOLD = 0.01
pos = {}
node_color,node_size = [],[]
edge_color,edge_width = [],[]
#P = []
EDGE_THICKNESS = 25
pheromone_thickness = 1
ant_thickness = 25

INIT_WEIGHT_FACTOR = 20
MAX_PATH_LENGTH = 38

FRAME_INTERVAL = 1000


global EXPLORE_CHANCES
global EXPLORES
EXPLORE_CHANCES = 0
EXPLORES = 0

DRAW_AND_QUIT = False

DEBUG = False

DEBUG_QUEUES = False
DEBUG_PATHS = False

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

def clear_queues(G):
    '''
    empties all node and edge queues
    '''
    for u in G.nodes_iter():
        G.node[u]['queue'] = []
    
    for u, v in G.edges():
        G[u][v]['forwards_queue'] = []
        G[u][v]['backwards_queue'] = []

def init_graph(G):
    '''
    initializes the graph for drawing
    
    sets all edge weights to 0, and creates mapping between each node and an integer, and
    between each edge and an integer
    '''
    
    # map each node to an integer
    for i,u in enumerate(sorted(G.nodes())):
        M[i] = u
        Minv[u] = i    
        
        if 'road' not in G.graph['name'] and G.graph['name'] != 'subelji':
            pos[u] = [u[0],u[1]] # position is the same as the label.
        
        G.node[u]['queue'] = []

        if u in G.graph['nests']:
            node_size.append(100)
            node_color.append('r')
        else:
            node_size.append(10)
            node_color.append('k')
            
            
    # map each edge to an integer
    # since this is an undirected graph, both orientations of an edge map to same integer
    # each integer maps to one of the two orientations of the edge
    for i, (u,v) in enumerate(sorted(G.edges())):
        G[u][v]['weight'] = MIN_PHEROMONE
        
        G[u][v]['forwards'] = tuple(sorted((u, v)))
        G[u][v]['forwards_queue'] = []
        
        G[u][v]['backwards'] = tuple(sorted((u, v))[::-1])
        G[u][v]['backwards_queue'] = []
        
        Ninv[(u, v)] = i
        N[i] = (u, v)        
        Ninv[(v, u)] = i
        
        init_path = G.graph['init_path']
        if (u, v) in init_path:
            edge_color.append('g')
            edge_width.append(10)
        elif (v, u) in init_path:
            edge_color.append('g')
            edge_width.append(10)
        else:
            edge_color.append('k')
            edge_width.append(1)
            
    G.graph['node_map'] = M
    G.graph['node_map_inv'] = Minv
    G.graph['edge_map'] = N
    G.graph['edge_map_inv'] = Ninv
    
def color_graph(G, c, w, figname, cost=None):
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
    unique_weights = set()
    for u, v in G.edges():
        index = None
        try:
            index = Ninv[(u, v)]
        except KeyError:
            index = Ninv[(v, u)]
        wt = G[u][v]['weight']
        width = 0
        # don't draw edges with miniscule weight
        if wt > AFTER_GRAPH_THRESHOLD:
            width = 1 + (wt * w * EDGE_THICKNESS)
            colors[index] = c
        else:
            width = 1
            colors[index] = 'k'
        
        widths[index] = width
        unique_weights.add(wt)

    if 'road' in G.graph['name'] or G.graph['name'] == 'subelji':
        nx.draw(G, with_labels=False, node_size=node_size, edge_color=colors,\
            node_color=node_color, width=widths, nodelist = sorted(G.nodes()), \
            edgelist = sorted(G.edges()))
    else:
        nx.draw(G, pos=pos, with_labels=False, node_size=node_size, edge_color=colors,\
            node_color=node_color, width=widths, nodelist = sorted(G.nodes()), \
            edgelist = sorted(G.edges()))
    PP.draw()
    #PP.show()
    PP.savefig(figname + '.png', format='png')
    PP.close()
    
    os.system('convert %s.png %s.pdf' % (figname, figname))

def pheromone_subgraph(G, origin=None, destination=None):
    '''
    Generates the subgraph induced by the edges with pheromone
    
    G - graph with pheromone edges
    
    origin, destination - nest and target vertices that should be included even if not
    adjacent to a pheromone edge
    '''
    G2 = nx.Graph()
    for u, v in G.edges_iter():
        # need enough pheromone for the ant to be able to detect it on that edge
        wt = G[u][v]['weight']
        if wt > MIN_DETECTABLE_PHEROMONE:
            G2.add_edge(u, v)
            G2[u][v]['weight'] = wt
            
    # add the nests to the graph even if not connected anywhere
    if origin not in G2:
        G2.add_node(origin)
    if destination not in G2:
        G2.add_node(destination)
    return G2
    
def pheromone_paths(G, origin, destination, limit=15):
    '''
    computes all paths between the origin and destination that use only pheromone edges
    does this by enumerating all paths in the pheromone subgraph
    
    G - graph with pheromone edges
    
    origin, destination - vertices between which to find all paths
    '''
    G2 = pheromone_subgraph(G, origin, destination)
    return list(nx.all_simple_paths(G2, origin, destination, limit))

def has_pheromone_path(G, origin, destination):
    '''
    Checks whether there exists a pheromone path between origin and destination.  Does
    this by checking there is a path between origin and destination in the pheromone
    subgraph.
    
    G - graph with pheromone edges
    
    origin, destination - vertices between which to check for a path
    '''
    G2 = pheromone_subgraph(G, origin, destination)
    return nx.has_path(G2, origin, destination)

def count_nonzero(G, curr):
    '''
    Counts the number of neighboring edges with pheromone above the detection threshold
    '''
    count = 0
    for neighbor in G.neighbors(curr):
        if G[curr][neighbor]['weight'] > MIN_DETECTABLE_PHEROMONE:
            count += 1
    return count
    
def pheromone_cost(G):
    '''
    Counts the total number of pheromone edges in the graph G
    '''
    G2 = nx.Graph()
    for u, v in G.edges_iter():
        if G[u][v]['weight'] > MIN_DETECTABLE_PHEROMONE:
            G2.add_edge(u, v)
    return G2.number_of_edges()
    
def path_prob(G, path, explore_prob, strategy='uniform'):
    '''
    computes the probability of the ant taking a particular path on graph G according to
    the edge weights, explore probability, and choice function
    
    G - graph which ant traverses
    
    path - a path on G which the ant traverses
    
    explore_prob - probability that the ant takes an explore step
    
    strategy - the choice function the ant uses to determine its next step
    '''
    prob = 1
    prev = None
    for i in xrange(len(path) - 1):
        source = path[i]
        dest = path[i + 1]
        prob *= choice_prob(G, source, dest, explore_prob, prev, strategy)
        if prob == 0:
            break
        prev = source
    return prob
    
def path_prob_no_explore(G, path, strategy='uniform'):
    '''
    computes the probability of an ant taking a particular path on a graph G when explore
    steps are not allowed
    '''
    return path_prob(G, path, 0, strategy)

def pruning_plot(costs, figname, max_cost=None):
    '''
    plots the edge cost over time
    '''
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
    
def pheromone_dead_end(G, curr, prev):
    '''
    Checks if an ant is at a dead end.  We define a dead end as such: if the only edge
    with pheromone is the edge that the ant traversed on the previous step, then the ant
    is at a dead end, i.e. following pheromone did not allow the ant to reach the nest
    '''
    for n in G.neighbors(curr):
        if n != prev and G[curr][n]['weight'] > MIN_DETECTABLE_PHEROMONE:
            return False
    return True

def walk_to_path(walk):
    '''
    Converts a walk on a graph to a path by removing all cycles in the path
    '''
    assert len(walk) >= 2
    if walk[0] == walk[-1]:
        print walk
    assert walk[0] != walk[-1]
    path = []
    
    '''
    path does not have repeated vertices, so keep track of visited vertices on walk
    maps each vertex to the first position in the walk at which that vertex was
    visited
    '''
    visited = {}
    for i, node in enumerate(walk):
        if node not in visited:
            # new node, goes into the path
            path.append(node)
            visited[node] = len(path) - 1
        else:
            '''
            visited node, meaning our walk has cycled back to a vertex.  Thus we excise 
            all vertices that are part of that cycle from the path.
            '''
            prev = visited[node]
            for j in xrange(prev + 1, len(path)):
                del visited[path[j]]
            path = path[:prev + 1]
    assert len(path) >= 2
    assert path[0] == walk[0]
    assert path[-1] == walk[-1]
    
    return path

def queue_ant(G, queue_node, ant):
    '''
    adds an ant to the queue at a particular node in the graph
    '''
    G.node[queue_node]['queue'].append(ant)
            
def check_queues(G, queued_nodes, queued_edges, num_ants, verbose=False):
    '''
    Checks whether the queued nodes are correct.  Checks that each queued node has a 
    non-empty queue; checks that every ant appears in one of the queues, and that no
    ant appears in more than one queue
    '''
    queued_ants = []
    for queued_node in sorted(queued_nodes):
        queue = G.node[queued_node]['queue']
        if verbose:
            print "queued node", queued_node, queue
        assert len(G.node[queued_node]['queue']) > 0
        for ant in queue:
            assert ant not in queued_ants
        queued_ants += list(queue)
    
    for edge_id in sorted(queued_edges):
        u, v = N[edge_id]
        directions = ['forwards', 'backwards']
        total_size = 0
        for direction in directions:
            queue = G[u][v][direction + '_queue']
            if verbose:
                print "queued edge " + direction, edge_id, (u, v), queue
            total_size += len(queue)
            for ant in queue:
                assert ant not in queued_ants
            queued_ants += list(queue)
        assert total_size > 0
    
    if verbose:
        if len(queued_ants) != num_ants:
            print len(queued_ants), sorted(queued_ants), num_ants
    assert len(queued_ants) == num_ants

def path_to_edges(path):
    '''
    converts a path represented as a series of vertices to a path represented as a series
    of edges
    '''
    edges = []
    for i in xrange(len(path) - 2):
        edges.append(Ninv[(path[i], path[i + 1])])
    return edges

def wasted_edges(G, useful_edges):
    '''
    count the number of edges that have pheromone but are not part of the network that the
    ants use.
    
    Assumes useful edges is a list of edge ids, rather than actual edges
    '''
    wasted_edges = 0
    wasted_edge_weight = 0
    for u, v in G.edges_iter():
        wt = G[u][v]['weight']
        if wt > MIN_DETECTABLE_PHEROMONE:
            edge_id = Ninv[(u, v)]
            if edge_id not in useful_edges:
                wasted_edges += 1
                wasted_edge_weight += wt
    return wasted_edges, wasted_edge_weight

def max_neighbors(G, source, prev=None):
    '''
    Gets all neighbors that are tied for the highest weight among all neighbors.  Ignores
    ants most previously visited vertex
    '''
    candidates = G.neighbors(source)
    # ignore previous vertex, ant won't consider it
    if prev != None:
        assert prev in candidates
        candidates.remove(prev)
    max_wt = float("-inf")
    max_neighbors = []
    for candidate in candidates:
        wt = G[source][candidate]['weight']
        if wt <= MIN_DETECTABLE_PHEROMONE:
            continue
        elif wt > max_wt:
            max_wt = wt
            max_neighbors = [candidate]
        elif wt == max_wt:
            max_neighbors.append(candidate)
    return max_neighbors
    
def maximal_paths(G, source, dest, limit=None):
    queue = [[source]]
    max_paths = []
    while len(queue) > 0:
        curr_path = queue.pop(0)
        if limit != None and len(curr_path) > limit:
            continue
        curr = curr_path[-1]
        prev = None
        if len(curr_path) > 1:
            prev = curr_path[-2]
        max_n = max_neighbors(G, curr, prev)
        for max_neighbor in max_n:
            if max_neighbor not in curr_path:
                new_path = curr_path + [max_neighbor]
                if max_neighbor == dest:
                    max_paths.append(new_path)
                else:
                    queue.append(new_path)
    return max_paths

def print_queues(G):
    for u in sorted(G.nodes()):
        if len(G.node[u]['queue']) > 0:
            print u, G.node[u]['queue']
        
    for u, v in sorted(G.edges()):
        if len(G[u][v]['forwards_queue']) + len(G[u][v]['backwards_queue']) > 0:
            print u, v
        if len(G[u][v]['forwards_queue']) > 0:
            print 'forwards', G[u][v]['forwards_queue']
        if len(G[u][v]['backwards_queue']) > 0:
            print 'backwards', G[u][v]['backwards_queue']

def check_path(G, path):
    for i in xrange(len(path) - 1):
        u, v = path[i], path[i + 1]
        if u != v:
            assert G.has_edge(u, v)
    
def repair(G, pheromone_add, pheromone_decay, explore_prob, strategy='uniform', \
            num_ants=100, max_steps=10000, num_iters=1, print_graph=False, video=False, \
            nframes=200, video2=False, cost_plot=False, backtrack=False, \
            decay_type='linear', node_queue_lim=1, edge_queue_lim=1, one_way=False):
    """ """
    
    graph_name = G.graph['name']
    nests = G.graph['nests']
    
    out_items = ['repair', strategy, graph_name, decay_type]
    if backtrack:
        out_items.append('backtrack')
    if one_way:
        out_items.append('one_way')
    out_str = '_'.join(out_items)
    
    def next_destination(origin):
        idx = nests.index(origin)
        idx += 1
        idx %= len(nests)
        return nests[idx]
    
    num_edges = G.size()
    
    nframes = min(nframes, max_steps)
    
    pher_str = "%d, %f, %f, " % (num_ants, explore_prob, pheromone_decay)
    data_file = open('ant_%s%d.csv' % (out_str, max_steps), 'a')
    pruning_file = open('ant_%s_pruning.csv' % out_str, 'a')

    if video:
        fig = PP.figure()
    
    init_path = G.graph['init_path']
    
    for iter in xrange(num_iters):    
        nonzero_edges = set()
        for u, v in G.edges_iter():
            G[u][v]['weight'] = MIN_PHEROMONE
            G[u][v]['units'] = []
        for u, v in init_path:
            G[u][v]['weight'] += pheromone_add * INIT_WEIGHT_FACTOR
            if decay_type == 'linear':
                G[u][v]['units']+= [pheromone_add] * INIT_WEIGHT_FACTOR
            nonzero_edges.add(Ninv[(u, v)])
        clear_queues(G)
    
        if iter == 0 and print_graph:
            pass #color_graph(G, 'g', pheromone_thickness, "graph_before")
        print str(iter) + ": " + pher_str
        explore = defaultdict(bool)
        prevs = {}
        prevs2 = {}
        prevs3 = {}
        currs = {}
        paths = {}
        walks = {}
        destinations = {}
        origins = {}
        edge_weights = defaultdict(list)
        deadend = {}
        search_mode = defaultdict(lambda: False)
        costs = []
    
        path_counts = defaultdict(int)
        chosen_walk_counts = defaultdict(int)
        max_entropy = None
        max_walk_entropy = None
            
        connect_time = -1
    
        queued_nodes = set()
        queued_edges = set()
            
        for ant in xrange(num_ants):
            origin = nests[ant % len(nests)]
            origins[ant] = origin
            destinations[ant] = next_destination(origin)
            prev_curr = None
            if one_way:
                prev, curr = init_path[choice((len(init_path) // 2))]
            else:
                prev, curr = init_path[choice(len(init_path))]
            if random() <= 0.5:
                prev, curr = curr, prev
            
            paths[ant] = [prev, curr]
            if destinations[ant] in paths[ant]:
                origins[ant], destinations[ant] = destinations[ant], origins[ant]
            walks[ant] = [prev, curr]
            prevs[ant] = prev
            prevs2[ant] = None
            prevs3[ant] = None
            currs[ant] = curr
            deadend[ant] = False
            queue_ant(G, curr, ant)
            queued_nodes.add(curr)
    
        steps = 1
        rounds = 1
        max_weight = MIN_PHEROMONE
        unique_weights = set()
        max_cost = 0
        costs = []
        curr_entropy = None
        curr_walk_entropy = None
    
    
        for u, v in sorted(G.edges()):
            index = Ninv[(u, v)]
            wt = G[u][v]['weight']
            unique_weights.add(wt)
            max_weight = max(max_weight, wt)
            if wt <= MIN_DETECTABLE_PHEROMONE:
                edge_weights[index].append(None)
            else:
                edge_weights[index].append(wt)
    
        critical_edges_file = None
        if graph_name == 'minimal':
            critical_edges_file = open('critical_edges.csv', 'a')
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
                        
            if graph_name == 'minimal':
                w1 = G[(1, 3)][(2, 3)]['weight']
                w2 = G[(1, 3)][(1, 4)]['weight']
                critical_str = '%d, %f, %f\n' % (steps, w1, w2)
                critical_edges_file.write(critical_str)
            
            G2 = G.copy()
        
            empty_nodes = set()
            new_queue_nodes = set()
            moved_ants = set()
            
            empty_edges = set()
        
            updated_paths_ants = set()
            
            if DEBUG_QUEUES:
                check_queues(G2, queued_nodes, queued_edges, num_ants)
            
            for queued_node in queued_nodes:
                queue = G2.node[queued_node]['queue']
                #queue = G2.node[node]['queue']
                #assert len(queue) > 0
            
                qlim = node_queue_lim
                if qlim == -1:
                    qlim = len(queue)
            
                next_ants = []
                q = 0
                while q < len(queue) and len(next_ants) < qlim:
                    if queue[q] not in moved_ants:
                        q_ant = queue[q]
                        next_ants.append(q_ant)
                    q += 1
                
            
                for next_ant in next_ants:
                    queue.remove(next_ant)

            
                for queued_ant in queue:
                    if queued_ant not in moved_ants:
                        moved_ants.add(queued_ant)
                        paths[queued_ant].append(queued_node)
                        if DEBUG_PATHS:
                            check_path(G, paths[queued_ant])
                        walks[queued_ant].append(queued_node)
                        path = paths[queued_ant]
            
                if len(queue) == 0:
                    empty_nodes.add(queued_node)
            
                for next_ant in next_ants:
                    moved_ants.add(next_ant)          
            
                    curr = currs[next_ant]
                    prev = prevs[next_ant]
                    

                    if curr != origins[next_ant] and (not search_mode[next_ant])\
                                                 and pheromone_dead_end(G, curr, prev):
                        search_mode[next_ant] = True
            
                    n = G.neighbors(curr)
                    if curr != prev and prev != None:
                        n.remove(prev)
                    if len(n) == 0:
                        deadend[next_ant] = (curr not in nests)
                        #print curr, deadend[j]
                    elif len(n) > 1:
                        deadend[next_ant] = False
                
                    if (prev == curr) or (curr == origins[next_ant] and not search_mode[next_ant]):
                        prev = None

                    
                    exp_prob = explore_prob
                    if search_mode[next_ant]:
                        exp_prob = explore_prob
                    elif (curr == origins[next_ant] and not search_mode[next_ant]):
                        exp_prob = 0
                    
                    
                    next, ex = next_edge(G, curr, exp_prob, strategy, prev, \
                                         destinations[next_ant], search_mode[next_ant],\
                                         backtrack)
                                                                   
                    if ex:
                        queue_ant(G2, curr, next_ant)
                        if not deadend[next_ant]:
                            G2[curr][next]['weight'] += 2 * pheromone_add
                            if decay_type == 'linear':
                                G2[curr][next]['units'].append(2 * pheromone_add)
                            nonzero_edges.add(Ninv[(curr, next)])
                        new_queue_nodes.add(curr)
                        empty_nodes.discard(curr)
                        prevs[next_ant] = curr
                        currs[next_ant] = curr
                        paths[next_ant].append(curr)
                        walks[next_ant].append(curr)
                        if DEBUG_PATHS:
                            check_path(G, paths[next_ant])
                    else:
                        if (curr, next) == G[curr][next]['forwards']:
                            G2[curr][next]['forwards_queue'].append(next_ant)
                        else:
                            G2[curr][next]['backwards_queue'].append(next_ant)
                        queued_edges.add(Ninv[(curr, next)])
                        
            queued_nodes.difference_update(empty_nodes)
            queued_nodes.update(new_queue_nodes)
            
            if DEBUG_QUEUES:
                check_queues(G2, queued_nodes, queued_edges, num_ants)
            
            for edge_id in queued_edges:
                u, v = N[edge_id]
                resulting_size = 0
                for direction in ['forwards', 'backwards']:
                    i = 0
                    curr, next = G2[u][v][direction]
                    edge_queue = G2[u][v][direction + '_queue']
                    
                    eqlim = edge_queue_lim
                    if edge_queue_lim == -1:
                        eqlim = len(edge_queue)
                    while len(edge_queue) > 0 and i < eqlim:
                        next_ant = edge_queue.pop(0)
                        i += 1                     
                        queue_ant(G2, next, next_ant)
                        
                        new_queue_nodes.add(next)
                        empty_nodes.discard(next)
                        prevs[next_ant] = curr
                        currs[next_ant] = next
                        if not deadend[next_ant]:
                            G2[curr][next]['weight'] += pheromone_add
                            if decay_type == 'linear':
                                G2[curr][add_neighbor]['units'].append(pheromone_add)
                            nonzero_edges.add(Ninv[(curr, next)])
                        
                        paths[next_ant].append(next)
                        walks[next_ant].append(next)

                        if DEBUG_PATHS:
                            check_path(G, paths[next_ant])
        
        
                        if next == destinations[next_ant]:
                            orig, dest = origins[next_ant], destinations[next_ant]
                            origins[next_ant], destinations[next_ant] = dest, orig
                            search_mode[next_ant] = False
            
                            walk = walks[next_ant]
                            chosen_walk_counts[tuple(walk)] += 1
            
                            path = walk_to_path(walk)
                            start = path[0]
                            end = path[-1]
                            idx1 = nests.index(orig)
                            idx2 = nests.index(dest)
                            if idx2 > idx1:
                                path = path[::-1]
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
            
                            walks[next_ant] = [origins[next_ant]]
            
                        elif next == origins[next_ant]:
                            search_mode[next_ant] = True
                            
                    for j in xrange(len(edge_queue)):
                        next_ant = edge_queue[j]
                        paths[next_ant].append(curr)
                        if DEBUG_PATHS:
                            check_path(G, paths[next_ant])
                        walks[next_ant].append(curr)
                        resulting_size += 1
                
                if resulting_size == 0:
                    empty_edges.add(edge_id)
                        
            
            queued_nodes.difference_update(empty_nodes)
            queued_nodes.update(new_queue_nodes)
            queued_edges.difference_update(empty_edges)
            
            if DEBUG_QUEUES:
                check_queues(G2, queued_nodes, queued_edges, num_ants)
            
            decay_func = get_decay_func_edges(decay_type)
            zero_edges = decay_func(G2, nonzero_edges, pheromone_decay, time=1, min_pheromone=MIN_PHEROMONE)
            nonzero_edges.difference_update(zero_edges)
        
            G = G2
        
            for u, v in sorted(G.edges()):
                index = Ninv[(u, v)]
                wt = G[u][v]['weight']
                unique_weights.add(wt)
                max_weight = max(max_weight, wt)
                if wt <= MIN_DETECTABLE_PHEROMONE:
                    edge_weights[index].append(None)
                else:
                    edge_weights[index].append(wt)
        
            if connect_time == -1 and has_pheromone_path(G, nests[0], nests[1]):
                connect_time = steps
        
            steps += 1
    
        cost = 0
        max_wt = 0
        for u, v in G.edges_iter():
            wt = G[u][v]['weight']
            if wt > MIN_PHEROMONE:
                cost += 1
            max_wt = max(wt, max_wt)
                            
        e_colors = edge_color[:]
        e_widths = edge_width[:]
        n_colors = node_color[:]
        n_sizes = node_size[:]
    
        for nest in nests:
            n_colors[Minv[nest]] = 'm'
            n_sizes[Minv[nest]] = 100
                            
        def init():
            nx.draw(G, pos=pos, with_labels=False, node_size=n_sizes, edge_color=e_colors,\
                    node_color=n_colors, width=e_widths, nodelist = sorted(G.nodes()), \
                    edgelist = sorted(G.edges()))
    
        def redraw(frame):
            PP.clf()
            print frame
        
            n_colors = ['r'] * len(node_color)
            n_sizes = [10] * len(node_size)
        
            ax = PP.gca()
        
            for n in xrange(num_ants):             
                node = paths[n][frame + 1]
                index = Minv[node]
                n_colors[index] = 'k'
                n_sizes[index] += ant_thickness
                                        
            #if frame > 0:
            #    frame -= 1
            max_units = max_weight / pheromone_add
        
            e_colors = []
            e_widths = []
            for u, v in sorted(G.edges()):
                index = Ninv[(u, v)]
                edge_wt = edge_weights[index][frame]
                if edge_wt == None:
                    e_colors.append('k')
                    e_widths.append(1)
                else:
                    e_colors.append('g')
                    e_widths.append(1 + 25 * (edge_wt / max_weight))
        
            for nest in nests:
                n_colors[Minv[nest]] = 'm'
                #n_sizes[Minv[nest]] = min(n_sizes[Minv[nest]], 100)
                #print nest, n_sizes[Minv[nest]]

            nx.draw(G, pos=pos, with_labels=False, node_size=n_sizes, edge_color=e_colors, \
                    node_color=n_colors, width=e_widths, nodelist = sorted(G.nodes()), \
                    edgelist = sorted(G.edges()))
            f = PP.draw()
            return f,
    
        if nframes == -1:
            nframes = steps
    
        if video:    
            ani = animation.FuncAnimation(fig, redraw, init_func=init, frames=nframes, \
                                          interval = FRAME_INTERVAL)
            ani.save("ant_" + out_str + str(iter) + ".mp4")
        
        if print_graph:        
            color_graph(G, 'g', (pheromone_add / max_wt), "graph_after_%s%d_e%0.2fd%0.2f" \
                        % (out_str, max_steps, explore_prob, pheromone_decay), cost)
            print "graph colored"
    
        costs.append(cost)
        max_cost = max(max_cost, cost)
        costs = PP.array(costs)
        pruning = (max_cost - cost) / float(max_cost)
        if cost_plot:
            figname = "pruning/pruning_%s%d_e%0.2fd%0.2f" % (out_str, max_steps, \
                       explore_prob, pheromone_decay, cost)
            pruning_plot(costs, figname, max_cost)
                
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
    
        nest, target = nests[0], nests[1]
            
        has_path = has_pheromone_path(G, nest, target)
        after_paths = []
        if has_path:
            if False:#'uniform' in strategy:
                after_paths = pheromone_paths(G, nest, target, MAX_PATH_LENGTH)
            else:
                after_paths = maximal_paths(pheromone_subgraph(G), nest, target)
        path_probs = []
    
        useful_edges = set()
        path_lenghts = []
        for path in after_paths:
            path_prob = path_prob_no_explore(G, path, strategy)
            if path_prob > 0:
                path_probs.append(path_prob)
                edges = path_to_edges(path)
                useful_edges.update(edges)
                path_lengths.append(len(path))
        wasted_edge_count, wasted_edge_weight = wasted_edges(G, useful_edges)
    
        path_etr = None
        if len(path_probs) > 0:
            path_etr = entropy(path_probs)
            path_etr = abs(path_etr)
        else:
            has_path = False
            
        mean_path_len = None
        if len(path_lengths) > 0:
            mean_path_len = mean(path_lengths)
        
        print "has path", has_path
        print "path entropy", path_etr
        
        journey_times = []
        journey_lengths = []
        walk_counts = defaultdict(int)
        total_steps = 0
    
        write_items = [int(has_path), cost]
    
        if (path_etr != None) and path_etr != float("-inf") and (not PP.isnan(path_etr)):
            write_items.append(path_etr)
        else:
            write_items.append('')
    
        if len(walk_counts.values()) > 0:
            write_items += [walk_entropy, mean_journey_time, med_journey_time, walk_success_rate]
        else:
            write_items += ['', '', '', '']
    
        #write_items.append(walk_success_rate)
    
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
        write_items += [wasted_edge_count, wasted_edge_weight]
        
        if mean_path_len != None:
            write_items.append(mean_path_len)
        else:
            write_items.append('')
    
        ant_str = ', '.join(map(str, write_items))
        line = pher_str + ant_str + '\n'
        data_file.write(line)

def main():
    start = time.time()
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(asctime)s -- %(message)s'
    )
    
    '''
    graph_choices = ['fig1', 'full', 'simple', 'simple_weighted', 'simple_multi', \
                     'full_nocut', 'simple_nocut', 'small', 'tiny', 'medium', \
                     'medium_nocut', 'grid_span', 'grid_span2', 'grid_span3', 'er', \
                     'mod_grid', 'half_grid', 'mod_grid_nocut', 'half_grid_nocut',\
                     'mod_grid1', 'mod_grid2', 'mod_grid3', 'vert_grid', 'barabasi', \
                     'vert_grid1', 'vert_grid2', 'vert_grid3', 'caroad', 'paroad', 'txroad',
                     'subelji', 'minimal']
    '''
    decay_choices = ['linear', 'const', 'exp']
    

    usage="usage: %prog [options]"
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph", dest='graph', choices=GRAPH_CHOICES, default='full',\
                        help="graph to run algorithm on")
    parser.add_argument('-s', '--strategy', dest='strategy', choices=STRATEGY_CHOICES,\
                        default='rank', help="strategy to run")
    parser.add_argument("-x", "--repeats", type=int, dest="iterations", default=1,\
                        help="number of iterations") 
    parser.add_argument("-a", "--add", type=float, dest="pheromone_add",\
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
    parser.add_argument("-e", "--explore", type=float, dest="explore", default=0.05, \
                        help="explore probability")
    parser.add_argument("-m", "--max_steps", type=int, dest="max_steps", default=3000)
    parser.add_argument("-c", "--cost_plot", action="store_true", dest="cost_plot", default=False)
    parser.add_argument('-b', '--backtrack', action='store_true', dest='backtrack', default=False)
    parser.add_argument("-dt", "--decay_type", dest="decay_type", default="exp", \
                        choices=DECAY_CHOICES)
    parser.add_argument("-t", "--threshold", dest="threshold", type=float, default=0, \
                        help="minimum detectable pheromone threshold")
    parser.add_argument('-nql', '--node_queue_limit', type=int, dest='node_queue_lim', default=1)
    parser.add_argument('-eql', '--edge_queue_limit', type=int, dest='edge_queue_lim', default=1)
    parser.add_argument('-o', '--one_way', action='store_true', dest='one_way')

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
    decay_type = args.decay_type
    node_queue_lim = args.node_queue_lim
    edge_queue_lim = args.edge_queue_lim
    one_way = args.one_way

    # Build network.
    G = get_graph(graph)
    if G == None:
        return
    init_graph(G)
    
    if DRAW_AND_QUIT:
        nx.draw(G, pos=pos, with_labels=False, node_size=node_size, edge_color=edge_color, \
                node_color=node_color, width=edge_width, nodelist = sorted(G.nodes()), \
                edgelist = sorted(G.edges()))
        PP.draw()
        #print "show"
        #PP.show()
        PP.savefig("%s.png" % G.graph['name'], format='png')
        PP.close()
        return None

    # Run recovery algorithm.
    repair(G, pheromone_add, pheromone_decay, explore, strategy, num_ants, max_steps,\
               num_iters, print_graph, video, frames, video2, cost_plot, backtrack, \
               decay_type, node_queue_lim, edge_queue_lim, one_way)
    
    # =========================== Finish ============================
    logging.info("Time to run: %.3f (mins)" %((time.time()-start) / 60))


if __name__ == "__main__":
    main()
