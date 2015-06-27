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

#seed(10301949)

Minv = {} # node tuple -> node id
M = {}    # node id -> node tuple

N = {}    # edge -> edge id
Ninv = {} # edge id -> edge

MAX_STEPS= 5000
MIN_PHEROMONE = 0.1
pos = {}
node_color,node_size = [],[]
edge_color,edge_width = [],[]
P = []
path_thickness = 1.5
pheromone_thickness = 10
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
    #print len(unique_weights)
    nx.draw(G, pos=pos, with_labels=False, node_size=node_size, edge_color=colors, node_color=node_color, width=widths)
    PP.draw()
    #PP.show()
    PP.savefig(figname)
    PP.close()

def check_graph_weights(G):
    for u, v in G.edges_iter():
        wt = G[u][v]['weight']
        assert wt >= MIN_PHEROMONE

def decay_graph(G, decay):
    for u, v in G.edges_iter():
        wt = G[u][v]['weight']
        assert wt >= MIN_PHEROMONE
        x = max(MIN_PHEROMONE, wt - decay)
        assert wt >= MIN_PHEROMONE
        G[u][v]['weight'] = x

def get_weights(G, start, candidates):
    weights = map(lambda x : G[start][x]['weight'], candidates)
    return array(weights)
    
def rand_edge(G, start=None, candidates = None):
    if candidates == None: 
        assert start != None
        candidates = G.neighbors(start)
    weights = get_weights(G, start, candidates)
    weights = weights / float(sum(weights))
    next = candidates[choice(len(candidates),1,p=weights)[0]]
    return next

def max_edge(G, start = None, candidates=None):
    if candidates == None:
        assert start != None
        candidates = G.neighbors(start)
    weights = get_weights(G, start, candidates)
    next = argmax(weights)
    next = candidates[next]
    return next
    
def next_edge(G, start, explore_prob=0.1, prev=None):
    unexplored = []
    explored = []
    neighbors = G.neighbors(start)
    for neighbor in neighbors:
        wt = G[start][neighbor]['weight']
        if wt == MIN_PHEROMONE:
            unexplored.append(neighbor)
        else:
            explored.append(neighbor)
    flip = random()
    if flip < explore_prob and len(unexplored) > 0:
        next = choice(len(unexplored))
        next = unexplored[next]
        return next, True
    candidates = explored + unexplored
    if (not BACKTRACK) and (prev != None) and (len(candidates) > 1):
        assert prev in candidates
        candidates.remove(prev) 
    if len(explored) == 1:
        return explored[0], False
    elif MAX:
        return max_edge(G, start, candidates), False
    else:
        return rand_edge(G, start, candidates), False

def count_nonzero(G, curr):
    count = 0
    for neighbor in G.neighbors(curr):
        if G[curr][neighbor]['weight'] > MIN_PHEROMONE:
            count += 1
    return count

def deviate(G,num_iters, num_ants, pheromone_add, pheromone_decay, print_path=False, print_graph=False, video=False, nframes=200, explore=0.1):
    """ """
    # os.system("rm -f graph*.png")
    # Put ants at the node adjacent to e, at node (4,3).
    bkpt = (4,3)
    init = (5,3)
    target = (3,2)
    nest = (8,3)
    
    def next_destination(prev):
        if prev == target:
            return nest
        return target
    
    assert G.has_node(bkpt)
    num_edges = G.size()

    data_file = open('ant_deviate.csv', 'a')
    pher_str = "%d, %f, %f, " % (num_ants, explore, pheromone_decay)
    # Repeat 'num_iters' times 
    for iter in xrange(num_iters):
        if video:
            fig = PP.figure()
        for u, v in G.edges_iter():
            G[u][v]['weight'] = MIN_PHEROMONE
        for u, v in P:
            G[u][v]['weight'] += pheromone_add
        
        if iter == 0 and print_graph:
            color_graph(G, 'g', pheromone_thickness, "graph_before")
        
        explore = defaultdict(bool)
        paths = {}
        destinations = {}
        origins = {}
        hits = defaultdict(int)
        misses = defaultdict(int)
        edge_weights = defaultdict(list)
        
        # search = defaultdict(lambda : True)
        
        hit_counts0 = [0]
        miss_counts0 = [0]
        
        hit_counts1 = [0]
        miss_counts1 = [0]
        
        attempts = defaultdict(lambda : 1)
        hitting_times = defaultdict(int)
        success_lengths = defaultdict(list)
        
        for ant in xrange(num_ants):
            if ant % 2 == 0 or DEAD_END:
                if BREAK:
                    paths[ant] = [init, bkpt]
                else:
                    paths[ant] = [nest, (7, 3)]
                destinations[ant] = target
                origins[ant] = nest
            else:
                paths[ant] = [target, (3, 3)] 
                destinations[ant] = nest
                origins[ant] = target     
        i = 1
        max_weight = MIN_PHEROMONE
        unique_weights = set()
        while i <= MAX_STEPS:
            check_graph_weights(G)
            h0 = hit_counts0[-1]
            h1 = hit_counts1[-1]
            
            m0 = miss_counts0[-1]
            m1 = miss_counts1[-1]
                
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
            for inert in xrange(i, num_ants):
                paths[inert].append(paths[inert][-1])
            for j in xrange(min(num_ants, i)):
                curr = paths[j][-1]
                prev = paths[j][-2]
                if prev == curr:
                    prev = None
                if explore[j]:
                    paths[j].append(prev)
                    explore[j] = False
                    G[curr][prev]['weight'] += pheromone_add
                else:
                    next, ex = next_edge(G, curr, explore_prob=explore, prev=prev)
                    explore[j] = ex
                    paths[j].append(next)
                    G[curr][next]['weight'] += pheromone_add
                    if next == destinations[j]:
                        hits[j] += 1
                        if origins[j] == nest:
                            h0 += 1
                        else:
                            h1 += 1
                        origins[j], destinations[j] = destinations[j], origins[j]
                        
                        success_lengths[j].append(i - hitting_times[j])
                        hitting_times[j] = i
                        attempts[j] += 1
                        
                    elif next == origins[j]:
                        if origins[j] == nest:
                            m0 += 1
                        else:
                            m1 += 1
                        misses[j] += 1
                        attempts[j] += 1
                                    
            decay_graph(G, pheromone_decay)
            
            pheromone_add = max(pheromone_add - ADD_PRUNE, MIN_ADD)
            
            hit_counts0.append(h0)
            hit_counts1.append(h1)
            
            miss_counts0.append(m0)
            miss_counts1.append(m1)
            
            i += 1
                    
        # e_colors = ['k'] * len(edge_color)
#         e_widths = [1] * len(edge_width)
#         n_colors = ['r'] * len(node_color)
#         n_sizes = [10] * len(node_size)
        
        e_colors = edge_color[:]
        e_widths = edge_width[:]
        n_colors = node_color[:]
        n_sizes = node_size[:]
        
        n_colors[Minv[target]] = 'm'
        n_colors[Minv[nest]] = 'y'
        
        n_sizes[Minv[target]] = n_sizes[Minv[nest]] = 100
        
        # print "path lengths"
#         for ant in xrange(num_ants):
#             print len(paths[ant])
#         
#         print "edge weight lengths"
#         for edge in edge_weights:
#             print len(edge_weights[edge])
#             
#         print "hit/miss count lengths"
#         print len(hit_counts0), len(hit_counts1), len(miss_counts0), len(miss_counts1)
                
        def init():
            nx.draw(G, pos=pos, with_labels=False, node_size=n_sizes, edge_color=e_colors, node_color=n_colors, width=e_widths)
        
        def redraw(frame):
            PP.clf()
            frame = min(frame, MAX_STEPS)
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
                h0 = hit_counts0[frame]
                m0 = miss_counts0[frame]
            
                h1 = hit_counts1[frame]
                m1 = miss_counts1[frame]
            
                uv_str = str(h0) + ' hits ' + str(m0) + ' misses'
                vu_str = str(h1) + ' hits ' + str(m1) + ' misses'
             
                PP.text(0.1, 0.9, 'nest1 -> nest2: ' + uv_str, transform=ax.transAxes, fontsize=7)
                PP.text(0.1, 0.88, 'nest2 -> nest1: ' + vu_str, transform=ax.transAxes, fontsize=7)
            
            if frame > 0:
                frame -= 1
                max_units = max_weight / pheromone_add
                for index in edge_weights:
                    wt = edge_weights[index][frame]
                    if wt != None:
                        units = edge_weights[index][frame]
                        e_widths[index] = 1 + 5 * (units / max_units)
                        e_colors[index] = 'g'
                    
            #print e_widths
                    
            n_colors[Minv[target]] = 'm'
            n_colors[Minv[nest]] = 'y'
            n_sizes[Minv[target]] = max(n_sizes[Minv[target]], 100)
            n_sizes[Minv[nest]] = max(n_sizes[Minv[nest]], 100)

            nx.draw(G, pos=pos, with_labels=False, node_size=n_sizes, edge_color=e_colors, node_color=n_colors, width=e_widths)
            f = PP.draw()
            return f,
        
        if video:    
            ani = animation.FuncAnimation(fig, redraw, init_func=init, frames=nframes, interval = 1000)
            ani.save("ant_deviate" + str(iter) + ".mp4")

        # Output results.
        path_lengths, revisits = [], []
        right_nest, wrong_nest = 0.0, 0.0
        hit_counts, miss_counts = [], []
        
        for k in xrange(num_ants):
            path = paths[k]
            revisits.append(len(path) - len(set(path)))
            path_lengths.append(len(path))
            h = hits[k]
            m = misses[k]
            hit_counts.append(h)
            miss_counts.append(m)
            if h > 0:
                right_nest += 1
            if m > 0:
                wrong_nest += 1
            top10 = (k + 1) <= 0.1 * num_ants
            bottom10 = (k + 1) >= 0.9 * num_ants
            mean_success_len = MAX_STEPS
            if len(success_lengths[j]) != 0:
                mean_success_len = mean(success_lengths[j])
            att = attempts[j]
            ant_str = "%d, %d, %d, %d, %d, %d, %.2f, %d\n" % (len(path), top10, bottom10, revisits[-1], hits[k], misses[k], mean_success_len, att)
            data_file.write(pher_str + ant_str)
            

        # Compare time for recovery for first 10% of ants with last 10%.
        if num_ants < 10:
            first_10, last_10 = -1, -1
        else:
            first_10 = mean(path_lengths[0:int(num_ants*0.1)])
            last_10  = mean(path_lengths[int(num_ants*0.9):])
        
        right_prop = right_nest / num_ants
        wrong_prop = wrong_nest / num_ants

        # Output results.
        assert len(path_lengths) == num_ants == len(revisits)
        print iter + 1

        if print_path:        
            for i in xrange(num_ants):
                path = paths[i]
                num_zeros = len(str(num_ants)) - len(str(i))
                fig_name = 'ant' + ('0' * num_zeros) + str(i)
                color_path(G, path, 'b', path_thickness, fig_name)
        
        if print_graph:        
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
    parser.add_option("-a", "--add", action="store", type="float", dest="pheromone_add", default=MIN_ADD,help="amt of phermone added")
    parser.add_option("-d", "--decay", action="store", type="float", dest="pheromone_decay", default=0.0,help="amt of pheromone decay")
    parser.add_option("-n", "--number", action="store", type="int", dest="num_ants", default=10,help="number of ants")
    parser.add_option("-p", "--print_path", action="store_true", dest="print_path", default=False)
    parser.add_option("-g", "--print_graph", action="store_true", dest="print_graph", default=False)
    parser.add_option("-v", "--video", action="store_true", dest="video", default=False)
    parser.add_option("-f", "--frames", action="store", type="int", dest="frames", default=200)
    parser.add_option("-e", "--explore", action="store", type="float", dest="explore", default=0.1)

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

    # Build network.
    G = fig1_network()

    #return

    # Run recovery algorithm.
    deviate(G,num_iters,num_ants,pheromone_add,pheromone_decay, print_path, print_graph, video, frames)
    
    #nx.draw(G,pos=pos,with_labels=False,node_size=node_size,edge_color=edge_color,node_color=node_color,width=edge_width)
    #PP.draw()
    #PP.show()
    #PP.savefig("fig1.pdf")
    #PP.close()

    
    # =========================== Finish ============================
    logging.info("Time to run: %.3f (mins)" %((time.time()-start) / 60))


if __name__ == "__main__":
    main()