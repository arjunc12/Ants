import networkx as nx
from random import choice
from sys import argv
from graphs import *

Minv = {} # node tuple -> node id
M = {}    # node id -> node tuple

Ninv = {}    # edge -> edge id
N = {}       # edge id -> edge

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
            
            
    # map each edge to an integer
    # since this is an undirected graph, both orientations of an edge map to same integer
    # each integer maps to one of the two orientations of the edge
    for i, (u,v) in enumerate(sorted(G.edges())):
        
        Ninv[(u, v)] = i
        N[i] = (u, v)        
        Ninv[(v, u)] = i


def walk_to_string(walk):
    walk_str = []
    for i in range(len(walk)):
        walk_str.append(str(Minv[walk[i]]))
    return '-'.join(walk_str)

def pure_random_walk(G, nest, target, ants):
    out_file = open('pure_rand_walk_%s.csv' % G.graph['name'], 'a')
    for ant in xrange(ants):
        curr = nest
        steps = 0
        walk = [curr]
        prev = None
        while curr != target:
            candidates = G.neighbors(curr)
            if prev in candidates and len(candidates) > 1:
                candidates.remove(prev)
            next = choice(candidates)
            prev = curr
            curr = next
            walk.append(next)
            steps += 1
        walk_str = walk_to_string(walk)
        out_file.write('%s, %d, %s\n' % (G.graph['name'], steps, walk_str))
    out_file.close()
    
if __name__ == '__main__':
    ants = int(argv[1])
    graphs = argv[2:]
    for graph in graphs:
        G = get_graph(graph)
        init_graph(G)
        nest, target = G.graph['nests']
        pure_random_walk(G, nest, target, ants)