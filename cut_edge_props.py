import matplotlib as mpl
mpl.use('agg')
import pylab
from collections import defaultdict
import pandas as pd
from sys import argv
import argparse

DATASETS_DIR = 'datasets/reformated_csv'

CUT_SHEETS = ['7b', '9a', '9c', '9d', '10', '11', '12', '13', '18', '19', '20', '21c', '22', '23b', '23c', '23e', '23f']

def edge_props2(sheet, cut_node=None):
    counts_file = '%s/reformated_counts%s.csv' % (DATASETS_DIR, sheet)
    df = pd.read_csv(counts_file, names = ['source', 'dest', 'time'], skipinitialspace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace=True)
    
    times = list(df['time'])
    deltas = []
    starttime = times[0]
    for time in times:
        deltas.append((time - starttime) / pylab.timedelta64(1, 's'))

    sources = list(df['source'])
    dests = list(df['dest'])
    counts = {}
    
    delta_edges = defaultdict(list)
    for i in xrange(len(deltas)):
        delta = deltas[i]
        source = sources[i]
        dest = dests[i]
        #edge = (source, dest)
        edge = tuple(sorted((source, dest)))
        delta_edges[delta].append(edge)
        counts[edge] = []

    for delta in sorted(delta_edges.keys()):
        for edge in counts:
            count = 0
            if len(counts[edge]) > 0:
                count = counts[edge][- 1]
            if edge in delta_edges[delta]:
                count += 1
            counts[edge].append(count)

    step_times = pylab.arange(1, len(delta_edges.keys()) + 1, dtype=pylab.float64)
    norms = pylab.zeros_like(step_times)
    for edge in counts:
        counts[edge] /= step_times
        norms += counts[edge]
 
    pylab.figure()
    for edge in counts:
        counts[edge] /= norms
        pylab.plot(step_times, counts[edge], label=edge)

    pylab.legend()
    pylab.savefig('cut_edge_props%s.pdf' % sheet, format='pdf')
    pylab.close()


def edge_props(sheet, cut_node=None):
    counts_file = '%s/reformated_counts%s.csv' % (DATASETS_DIR, sheet)
    counts = {}
    df = pd.read_csv(counts_file, names = ['source', 'dest', 'time'], skipinitialspace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace=True)
    times = list(df['time'])
    deltas = []
    starttime = times[0]
    
    sources = list(df['source'])
    dests = list(df['dest'])
    edges = []
    assert len(sources) == len(dests)
    for i in xrange(len(sources)):
        source, dest = sources[i], dests[i]
        edge = tuple(sorted((source, dest)))
        #edge = (source, dest)
        if cut_node == None or cut_node in edge:
            edges.append(edge)
            time = times[i]
            deltas.append((time - starttime) / pylab.timedelta64(1, 's'))
    
    counts = {}
    for edge in set(edges):
        counts[edge] = []

    for edge in edges:
        for e in counts:
            count = 0
            if len(counts[e]) > 0:
                count = counts[e][-1]
            if e == edge:
                count += 1
            counts[e].append(count)

    props = {}
    times = pylab.arange(1, len(edges) + 1, dtype=pylab.float64)
    for edge in counts:
        props[edge] = pylab.array(counts[edge]) / times

    pylab.figure()
    for edge in props:
        #pylab.plot(times, props[edge], label=edge)
        pylab.plot(deltas, props[edge], label=edge)
    pylab.legend()
    pylab.savefig('cut_edge_props%s.pdf' % sheet, format='pdf')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sh', '--sheets', default=None, nargs='+')
    parser.add_argument('-c', '--cut_node', default=None)

    args = parser.parse_args()
    sheets = args.sheets
    cut_node = args.cut_node

    if sheets == None:
        sheets = CUT_SHEETS

    for sheet in sheets:
        print sheet
        edge_props2(sheet, cut_node)

if __name__ == '__main__':
    main()
