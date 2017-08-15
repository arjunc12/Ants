import matplotlib as mpl
mpl.use('agg')
import pylab
from collections import defaultdict
import pandas as pd
from sys import argv

DATASETS_DIR = 'datasets/reformated_csv'

CUT_SHEETS = ['7b', '9a', '9c', '9d', '10', '11', '12', '13', '18', '19', '20', '21c', '22', '23b', '23c', '23e', '23f']

def edge_props(sheet):
    counts_file = '%s/reformated_counts%s.csv' % (DATASETS_DIR, sheet)
    counts = {}
    df = pd.read_csv(counts_file, names = ['source', 'dest', 'time'], skipinitialspace=True)
    sources = list(df['source'])
    dests = list(df['dest'])
    edges = []
    assert len(sources) == len(dests)
    for i in xrange(len(sources)):
        source, dest = sources[i], dests[i]
        #edges.append(tuple(sorted((source, dest))))
        edges.append((source, dest))

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
        pylab.plot(times, props[edge], label=edge)
    pylab.legend()
    pylab.savefig('cut_edge_props%s.pdf' % sheet, format='pdf')



def main():
    sheets = CUT_SHEETS
    if len(argv) > 1:
        sheets = argv[1]
    for sheet in sheets:
        print sheet
        edge_props(sheet)

if __name__ == '__main__':
    main()
