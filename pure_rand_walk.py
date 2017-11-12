import networkx as nx
from random import choice
from sys import argv
from graphs import *

Minv = {} # node tuple -> node id
M = {}    # node id -> node tuple

Ninv = {}    # edge -> edge id
N = {}       # edge id -> edge

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
    cycle_count = 0
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
            cycle_count += 1
            
    assert len(path) >= 2
    assert path[0] == walk[0]
    assert path[-1] == walk[-1]
    
    return path, cycle_count

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
    '''
    converts a sequences of nodes comprising a walk to a string. We can then output every different walk to a file
    to estimate the probabilities of all the different walks and estimate the entropy
    '''    
    walk_str = []
    for i in range(len(walk)):
        walk_str.append(str(Minv[walk[i]]))
    return '-'.join(walk_str)

def pure_random_walk(G, nest, target, ants):
    '''
    carry out an unweighted random walk. Each ant starts at the nest vertex and walks on G until reaching the target
    vertex.

    G - the graph on which to walk

    nest - the vertex from which each ant starts

    target - the vertex that each ant walks until reaching

    ants - the number of ants with which to perform an unweighted random walk
    '''
    out_file = open('pure_rand_walk_%s.csv' % G.graph['name'], 'a')
    for ant in xrange(ants):
        curr = nest
        steps = 0
        walk = [curr]
        prev = None
        while curr != target:
            candidates = G.neighbors(curr)
            # avoid taking the most recently visited vertex
            if prev in candidates and len(candidates) > 1:
                candidates.remove(prev)
            next = choice(candidates)
            prev = curr
            curr = next
            walk.append(next)
            steps += 1
        # output walk to file to be able to compute probabilities of different walks
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
