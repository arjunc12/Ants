import networkx as nx
from random import choice
from sys import argv

Minv = {} # node tuple -> node id
M = {}    # node id -> node tuple

N = {}    # edge -> edge id
Ninv = {} # edge id -> edge

def fig1_network():
    """ Manually builds the Figure 1 networks. """
    G = nx.grid_2d_graph(11,11)
    
    for i,u in enumerate(G.nodes_iter()):
        M[i] = u
        Minv[u] = i

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

    return G

def full_grid():
    G = nx.grid_2d_graph(11,11)
    G.remove_edge((4, 5), (5, 5))

    for i,u in enumerate(G.nodes_iter()):
        M[i] = u
        Minv[u] = i

    for i, (u,v) in enumerate(G.edges()):
        Ninv[(u, v)] = i
        N[i] = (u, v)        
        Ninv[(v, u)] = i
            
    for (u, v) in G.edges():
        assert (u, v) in Ninv
    return G

def walk_to_string(walk):
    walk_str = []
    for i in range(len(walk)):
        walk_str.append(str(Minv[walk[i]]))
    return '-'.join(walk_str)

def pure_random_walk(G, nest, target, ants):
    out_file = open('pure_rand_walk.csv', 'a')
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
        out_file.write('%d, %s\n' % (steps, walk_str))
    out_file.close()
    
if __name__ == '__main__':
    #G = fig1_network()
    G = full_grid()
    #target = (3,2)
    #nest = (8,3)
    target = (0,5)
    nest = (10,5)
    ants = int(argv[1])
    pure_random_walk(G, nest, target, ants)