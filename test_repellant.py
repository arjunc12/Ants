import networkx as nx
from random import random
from choice_functions import next_edge_rank
from decay_functions import decay_graph_exp
import numpy as np
from sys import argv

MAX_STEPS = float("inf")
INIT_WEIGHT = 10

def test_repellant(pluck_rate, explore_prob=0.2, decay=0.02, lay_back=True):
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G[1][2]['weight'] = INIT_WEIGHT
    G[1][3]['weight'] = 0
    
    steps = 0
    while G[1][2]['weight'] > G[1][3]['weight']:
        steps += 1
        #print "steps", steps, G[1][2]['weight'], G[1][3]['weight']
        next, ex = next_edge_rank(G, 1, explore_prob=explore_prob)
        if next == 3:
            G[1][3]['weight'] += 2
        else:
            G[1][2]['weight'] += 1
            pluck = random()
            if pluck >= pluck_rate and lay_back:
                G[1][2]['weight'] += 1
        decay_graph_exp(G, decay)
    return steps

def main():
    f = open('test_repellant.csv', 'a')
    for i in xrange(int(argv[1])):
        print i
        for pluck_rate in np.arange(0.99, 0.89, -0.01):
            steps = test_repellant(pluck_rate, lay_back=True)
            write_items = [steps, pluck_rate, 1]
            write_items = map(str, write_items)
            write_items = ', '.join(write_items)
            f.write('%s\n' % write_items)
        for pluck_rate in np.arange(0.01, 0.5, 0.01):
            steps = test_repellant(pluck_rate, lay_back=False)
            write_items = [steps, pluck_rate, 0]
            write_items = map(str, write_items)
            write_items = ', '.join(write_items)
            f.write('%s\n' % write_items)
        steps = test_repellant(0, lay_back=False)
        write_items = [steps, 0, 0]
        write_items = map(str, write_items)
        write_items = ', '.join(write_items)
        f.write('%s\n' % write_items)

if __name__ == '__main__':
    main()
