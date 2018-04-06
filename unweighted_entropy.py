from graphs import *
from sys import argv
from pure_rand_walk import walk_to_path, pure_random_walk
from collections import defaultdict
from scipy.stats import entropy
import argparse
import numpy as np
import pandas as pd

def bootstrap_entropy(G, ants=10000):
    nests = G.graph['nests']
    nest1, nest2 = nests[:2]
    path_lengths = []
    path_counts = defaultdict(int)
    for j in xrange(ants):
        walk = pure_random_walk(G, nest1, nest2)
        path, cycles = tuple(walk_to_path(walk))
        path = tuple(path)
        path_lengths.append(len(path))
        path_counts[path] += 1

    mean_path_len = np.mean(path_lengths)
    etr = entropy(path_counts.values())
    f = open('unweighted_entropy.csv', 'a')
    f.write('%s, %d, %f, %f\n' % (G.graph['name'], ants, mean_path_len, etr))
    f.close()

def mle_std(x):
    return np.std(x, ddof=1)

def entropy_stats():
    df = pd.read_csv('unweighted_entropy.csv', names=['graph', 'ants', 'mean_path_len', 'entropy'])
    df = df[['graph', 'entropy']]
    print df.groupby('graph').agg([np.mean, mle_std])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--ants', default=100, type=int)
    parser.add_argument('-x', '--num_iters', default=10000, type=int)
    parser.add_argument('-g', '--graphs', nargs='+', default=None)
    
    parser.add_argument('-b', '--bootstrap', action='store_true')
    parser.add_argument('-s', '--stats', action='store_true')

    args = parser.parse_args()
    
    ants = args.ants
    iters = args.num_iters
    graphs = args.graphs

    bootstrap = args.bootstrap
    stats = args.stats

    if bootstrap:
        for graph in graphs:
            for i in xrange(iters):
                G = get_graph(graph)
                if G != None:
                    bootstrap_entropy(G, ants)

    if stats:
        entropy_stats()

if __name__ == '__main__':
    main()
