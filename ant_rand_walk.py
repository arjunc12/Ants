#!/usr/bin/env python

from __future__ import division
import networkx as nx
import time,logging
from optparse import OptionParser
from matplotlib import pylab as PP
from numpy.random import seed,choice
from numpy import mean,median

#seed(10301949)

Minv = {} # node tuple -> node id
M = {}    # node id -> node tuple

N = {}    # edge -> edge id
Ninv = {} # edge id -> edge

MAX_STEPS=3000
pos = {}
node_color,node_size = [],[]
edge_color,edge_width = [],[]
P = []

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

    for u,v in G.edges_iter(): G[u][v]['weight'] = 1

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
        if u == (7,3) and v == (8,3): edge_color.append('r')
        elif u == (7,3) and v == (6,3): edge_color.append('r')
        elif u == (5,3) and v == (6,3): edge_color.append('r')
        elif u == (5,3) and v == (4,3): edge_color.append('r')
        elif u == (3,2) and v == (3,3): edge_color.append('r')
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


def run_recovery(G,num_iters,num_ants,pheromone_add,pheromone_decay):
    """ """
    
    # Put ants at the node adjacent to e, at node (4,3).
    bkpt = (4,3)
    init = (5,3)
    target = (3,2)
    nests = set([(8,3)])
    assert G.has_node(bkpt)
    num_edges = G.size()

    print "iter+1, num_ants, pheromone_add, pheromone_decay, mean(revisits), mean(path_lengths), median(path_lengths), len(wrong_nest), first_10,last_10)"


    # Repeat 'num_iters' times.
    for iter in xrange(num_iters):

        # Initialize all ants.
        paths = {}           # path traveled for each ant.
        wrong_nest = set()   # ants that revisit an old nest before recovering.
        for i in xrange(num_ants):
            paths[i] = [init,bkpt] # path traveled.
            next = paths[i][-1]

            j = 0
            while next != target:
                assert paths[i][0] == init

                # iterate through candidate edges and choose proportionally.
                curr,prev = paths[i][-1],paths[i][-2]
                candidates,weights = [],[]
                for neighbor in G.neighbors(curr):
                    if neighbor == prev: continue # history of last step.
                    candidates.append(Minv[neighbor])
                    weights.append(float(G[curr][neighbor]['weight']))

                # choose one proportionally.
                next = M[choice(candidates,1,p=[val/sum(weights) for val in weights])[0]]
                paths[i].append(next)

                # visited an old nest first.
                if next in nests: wrong_nest.add(i)

                # target found.
                if next == target: break 
                    
                # max reached, break.
                if len(paths[i]) == MAX_STEPS: break

            # Update pheromone on traversed edges (except the initial edge).
            for j in xrange(1,len(paths[i])-1):
                u,v = paths[i][j],paths[i][j+1]                
                assert G.has_edge(u,v)         
                G[u][v]['weight'] += pheromone_add

            # Decay weights.
            for u,v in G.edges_iter():                
                G[u][v]['weight'] = max(G[u][v]['weight']-pheromone_decay,0.1)

            assert G.size() == num_edges

        # todo: somehow the 'good edges' have to be preferentially reinforced. 

        # Compute statistics for each ant.
        revisits,path_lengths = [],[]
        for i in xrange(num_ants):
            revisits.append(len(paths[i]) - len(set(paths[i])))
            path_lengths.append(len(paths[i]))


        # Compare time for recovery for first 10% of ants with last 10%.
        first_10 = mean(path_lengths[0:int(num_ants*0.1)])
        last_10  = mean(path_lengths[int(num_ants*0.9):])

        # Output results.
        assert len(path_lengths) == num_ants == len(revisits)
        print "%i\t%i\t%.2f\t%.2f\t%i\t%i\t%i\t%i\t%i\t%i" %(iter+1,num_ants,pheromone_add,pheromone_decay,mean(revisits),mean(path_lengths),median(path_lengths),len(wrong_nest),first_10,last_10)
        
        first_path = paths[0]
        for i in xrange(len(first_path) - 1):
            edge = (first_path[i], first_path[i + 1])
            pos = None
            try:
                pos = Ninv[edge]
            except KeyError:
                pos = Ninv[(first_path[i + 1], first_path[i])]
            edge_color[pos] = 'b'
            edge_width[pos] = 2

                

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


    (options, args) = parser.parse_args()
    # ===============================================================

    # ===============================================================
    num_iters = options.iterations
    pheromone_add = options.pheromone_add
    pheromone_decay = options.pheromone_decay
    num_ants = options.num_ants

    # Build network.
    G = fig1_network()

    #return

    # Run recovery algorithm.
    run_recovery(G,num_iters,num_ants,pheromone_add,pheromone_decay)
    
    nx.draw(G,pos=pos,with_labels=False,node_size=node_size,edge_color=edge_color,node_color=node_color,width=edge_width)
    PP.draw()
    #PP.show()
    PP.savefig("fig1.pdf")
    PP.close()

    
    # =========================== Finish ============================
    logging.info("Time to run: %.3f (mins)" %((time.time()-start) / 60))


if __name__ == "__main__":
    main()