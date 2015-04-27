#!/usr/bin/env python

from __future__ import division
import networkx as nx
import time,logging
from optparse import OptionParser
from matplotlib import pylab as PP
from numpy.random import seed,choice
from numpy import mean,median, array
from collections import defaultdict
import os

#seed(10301949)

Minv = {} # node tuple -> node id
M = {}    # node id -> node tuple

N = {}    # edge -> edge id
Ninv = {} # edge id -> edge

MAX_STEPS= 3000
MIN_PHEROMONE = 0.1
pos = {}
node_color,node_size = [],[]
edge_color,edge_width = [],[]
P = []
path_thickness = 1.5
pheromone_thickness = 0.005
DEBUG_PATHS = False
OUTPUT_GRAPHS = False

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

def color_path(G, path, c, w, figname):
    """
    Given a path, colors that path on the graph and then outputs the colored path to a
    file
    
    TODO: make it clear what the order of edges taken is rather than just the edges taken
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
    colors, widths = edge_color[:], edge_width[:]
    unique_weights = set()
    for u, v in G.edges():
        index = None
        try:
            index = Ninv[(u, v)]
        except KeyError:
            index = Ninv[(v, u)]
        colors[index] = c
        wt = G[u][v]['weight']
        widths[index] = wt * w
        unique_weights.add(wt)
    print len(unique_weights)
    nx.draw(G, pos=pos, with_labels=False, node_size=node_size, edge_color=colors, node_color=node_color, width=widths)
    PP.draw()
    #PP.show()
    PP.savefig(figname)
    PP.close()

def decay_graph(G, decay):
    for u, v in G.edges_iter():
        wt = G[u][v]['weight']
        assert wt >= MIN_PHEROMONE
        x = max(MIN_PHEROMONE, wt - decay)
        G[u][v]['weight'] = x
    
def rand_edge(G, start, candidates = None):
    if candidates == None: 
        candidates = G.neighbors(start)
    weights = map(lambda x : G[start][x]['weight'], candidates)
    weights = array(weights)
    weights = weights / float(sum(weights))
    next = candidates[choice(len(candidates),1,p=weights)[0]]
    return next
    
def next_edge(G, start, prev=None):
    candidates = []
    neighbors = G.neighbors(start)
    for neighbor in neighbors:
        if neighbor == prev:
            continue
        wt = G[start][neighbor]['weight']
        if wt == MIN_PHEROMONE:
            return neighbor, True
        else:
            candidates.append(neighbor)
    assert prev not in candidates
    return rand_edge(G, start, candidates), False

def bfs(G,num_iters,num_ants,pheromone_add,pheromone_decay, print_path=False):
    """ """
    os.system("rm -f graph*.png")
    # Put ants at the node adjacent to e, at node (4,3).
    bkpt = (4,3)
    init = (5,3)
    target = (3,2)
    nests = set([(8,3)])
    assert G.has_node(bkpt)
    num_edges = G.size()

    print "iter+1, num_ants, pheromone_add, pheromone_decay, mean(revisits), mean(path_lengths), median(path_lengths), len(wrong_nest), first_10,last_10)"
    data_file = open('ant_bfs.csv', 'a')
    pher_str = "%d, %f, %f, " % (num_ants, pheromone_add, pheromone_decay)
    # Repeat 'num_iters' times 
    for iter in xrange(num_iters):
        
        for u, v in G.edges_iter():
            G[u][v]['weight'] = MIN_PHEROMONE
        for u, v in P:
            G[u][v]['weight'] += pheromone_add
        
        if iter == 0:
            color_graph(G, 'g', pheromone_thickness, "graph_before")
        
        at_nest = 0
        explore = defaultdict(bool)
        paths = defaultdict(lambda : [init, bkpt])
        wrong_nest = set()
        
        i = 1
        while (at_nest < num_ants) and i <= MAX_STEPS:
            for j in xrange(min(i, num_ants)):
                curr = paths[j][-1]
                prev = paths[j][-2]
                if curr != target:
                    if explore[j]:
                        paths[j].append(prev)
                        explore[j] = False
                        G[curr][prev]['weight'] += pheromone_add
                    else:
                        next, ex = next_edge(G, curr, prev)
                        explore[j] = ex
                        paths[j].append(next)
                        G[curr][next]['weight'] += pheromone_add
                        if next in nests:
                            wrong_nest.add(j)
                        if next == target:
                            at_nest += 1
                    
            decay_graph(G, pheromone_decay)
            
            if DEBUG_PATHS:
                print "----------------------"
                print i
                print "----------------------"
                for l in xrange(num_ants):
                    print l, paths[l]
            
            if OUTPUT_GRAPHS or i % 100 == 0:
                num_str = str(i)
                num_str = ('0' * (len(str(MAX_STEPS)) - len(num_str))) + num_str
                color_graph(G, 'g', pheromone_thickness, "graph_t" + num_str)
            i += 1
        

        # Output results.
        path_lengths, revisits = [], []
        for k in xrange(num_ants):
            path = paths[k]
            revisits.append(len(path) - len(set(path)))
            path_lengths.append(len(path))
            top10 = (k + 1) <= 0.1 * num_ants
            bottom10 = (k + 1) >= 0.9 * num_ants
            finished = path[-1] == target
            ant_str = "%d, %d, %d, %d, %d\n" % (len(path), top10, bottom10, revisits[-1], finished)
            data_file.write(pher_str + ant_str)
            

        # Compare time for recovery for first 10% of ants with last 10%.
        first_10 = mean(path_lengths[0:int(num_ants*0.1)])
        last_10  = mean(path_lengths[int(num_ants*0.9):])

        # Output results.
        assert len(path_lengths) == num_ants == len(revisits)
        print "%i\t%i\t%.2f\t%.2f\t%i\t%i\t%i\t%i\t%i\t%i" %(iter+1,num_ants,pheromone_add,pheromone_decay,mean(revisits),mean(path_lengths),median(path_lengths),len(wrong_nest),first_10,last_10)

        if print_path:        
            for i in xrange(num_ants):
                path = paths[i]
                num_zeros = len(str(num_ants)) - len(str(i))
                fig_name = 'ant' + ('0' * num_zeros) + str(i)
                color_path(G, path, 'b', path_thickness, fig_name)
                
        color_graph(G, 'g', pheromone_thickness, "graph_after_" + str(iter))
    
    data_file.close()
    

def main():
    start = time.time()
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(asctime)s -- %(message)s'
    )

    usage="usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-x", "--repeats", action="store", type="int", dest="iterations", default=10,help="number of iterations")
    parser.add_option("-a", "--add", action="store", type="float", dest="pheromone_add", default=0.0,help="amt of phermone added")
    parser.add_option("-d", "--decay", action="store", type="float", dest="pheromone_decay", default=0.0,help="amt of pheromone decay")
    parser.add_option("-n", "--number", action="store", type="int", dest="num_ants", default=10,help="number of ants")
    parser.add_option("-p", "--print", action="store_true", dest="print_path", default=False)

    (options, args) = parser.parse_args()
    # ===============================================================

    # ===============================================================
    num_iters = options.iterations
    pheromone_add = options.pheromone_add
    pheromone_decay = options.pheromone_decay
    num_ants = options.num_ants
    print_path = options.print_path

    # Build network.
    G = fig1_network()

    #return

    # Run recovery algorithm.
    bfs(G,num_iters,num_ants,pheromone_add,pheromone_decay, print_path)
    
    nx.draw(G,pos=pos,with_labels=False,node_size=node_size,edge_color=edge_color,node_color=node_color,width=edge_width)
    PP.draw()
    #PP.show()
    PP.savefig("fig1.pdf")
    PP.close()

    
    # =========================== Finish ============================
    logging.info("Time to run: %.3f (mins)" %((time.time()-start) / 60))


if __name__ == "__main__":
    main()