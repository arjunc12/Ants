import matplotlib as mpl
mpl.use('agg')
import pylab
import pandas as pd
from sys import argv
import argparse

def plot_edge_weights(graph_name, limits):
    df = pd.read_csv('critical_edges_%s.csv' % graph_name, header=None, names=['step', 'w1', 'w2'])

    limits = reversed(sorted(limits))
    for limit in limits:
        df2 = df[df['step'] <= limit]
        df2 = df.groupby('step', as_index=False).agg(pylab.mean)
        
        pylab.figure()
        pylab.plot(df2['step'], df2['w1'], c='b', label='dead end')
        pylab.plot(df2['step'], df2['w2'], c='r', label='path')
        #pylab.plot(df2['step'], df2['w1'] + df2['w2'], c='m', label='total')
        pylab.legend()
        pylab.xlabel('Simulation Steps')
        pylab.ylabel('edge weight')
        title_str = 'critical_edges_%s%d' % (graph_name, limit)
        pylab.savefig('%s.pdf' % title_str, format='pdf')
        pylab.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graphs', nargs='+')
    parser.add_argument('-l', '--limits', nargs='+', type=int)

    args = parser.parse_args()
    graphs = args.graphs
    limits = args.limits

    for graph in graphs:
        plot_edge_weights(graph, limits)

if __name__ == '__main__':
    main()
