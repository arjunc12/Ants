from random import random
from sys import argv
from choice_functions import next_edge_max
import networkx as nx
from ant_repair import decay_graph_exp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nsteps', dest='nsteps', type=int, required=True)
parser.add_argument('-e', '--explores', dest='explores', type=float, required=True, nargs='+')
parser.add_argument('-d', '--decays', dest='decays', type=float, required=True, nargs='+')
parser.add_argument('-x', '--iterations', dest='iterations', type=int)
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')

args = parser.parse_args()
nsteps = args.nsteps
explores = args.explores
decays = args.decays
iterations = args.iterations
verbose = args.verbose

G = nx.Graph()
G.add_edge(1, 2)
G.add_edge(1, 3)
f = open('theory.csv', 'a')

INIT_WEIGHT = 50

def print_weights(G):
    '''
    prints out weights of all edges in the graph
    '''
    for u, v in G.edges_iter():
        print (u, v), G[u][v]['weight']

for iter in xrange(iterations):
    for explore in explores:
        for decay in decays:
            G[1][2]['weight'] = INIT_WEIGHT
            G[1][3]['weight'] = 0
            
            choices1 = 0
            explores1 = 0
            choices2 = 0
            explores2 = 0
            choices3 = 0
            explores3 = 0
            
            for i in xrange(nsteps + 1):
                f.write('%f, %f, %d, %f, %f\n' % (explore, decay, i, G[1][2]['weight'], G[1][3]['weight']))
                if verbose:
                    print "---------------------"
                    print "before adding"
                    
                    print_weights(G)
                
                lower = True
                if G[1][3]['weight'] > G[1][2]['weight']:
                    lower = False
                
                if lower:
                    choices1 += 1
                else:
                    choices2 += 1
                choices3 += 1
                    
                next, ex = next_edge_max(G, 1, explore)
                
                if ex:
                    if lower:
                        explores1 += 1
                    else:
                        explores2 += 1
                    explores3 += 1
                
                assert next in [2, 3]
                add_amt = 2
                if next == 2:
                    add_amt -= 1
                G[1][next]['weight'] += add_amt
                if verbose:
                    print "next", next
                    print "after adding"
                    print_weights(G)

                decay_graph_exp(G, decay)
                
                if verbose:
                    print "after decaying"
                    print_weights(G)
                    print "---------------------"
            #print explore
            #print explores1, choices1, float(explores1) / float(choices1)
            #if choices2 > 0:
            #    print explores2, choices2, float(explores2) / float(choices2)
            #print explores3, choices3, float(explores3) / float(choices3)
                
                
    
f.close()