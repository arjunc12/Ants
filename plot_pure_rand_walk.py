import pandas as pd
import pylab
from collections import defaultdict
from scipy.stats import entropy
from sys import argv

graphs = argv[1:]
for graph in graphs:
    print graph
    df = pd.read_csv('pure_rand_walk_%s.csv' % graph, header=None, \
                     names = ['graph', 'steps', 'path'])
    #print max(df['steps'])
    #df2 = df[df['steps'] <= 1000]
    #print len(df2.index), len(df.index)
    pylab.figure()
    path_counts = defaultdict(int)
    N = 0
    for path in df['path']:
        N += 1
        path_counts[path] += 1
    etr = entropy(path_counts.values())
    pylab.hist(df['steps'])
    pylab.title("N = %d, mean = %0.2f, median = %d, entropy=%0.2f" % \
                (N, pylab.mean(df['steps']), pylab.median(df['steps']), etr))
    print 'mean', pylab.mean(df['steps'])
    print 'std', pylab.std(df['steps'], ddof=1)
    pylab.xlabel('Steps to reach target from nest')
    pylab.ylabel('Count')
    pylab.savefig('pure_rand_walk_%s.pdf' % graph, format='pdf')