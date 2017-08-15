import matplotlib as mpl
mpl.use('agg')
import pylab
import pandas as pd
from collections import defaultdict
from sys import argv

def critical_node_props(fname):
    times = defaultdict(list)
    counts = {}
    for line in open(fname):
        source, dest, time = line.split('), ')
        source += ')'
        dest += ')'
        time = int(time)
        times[time].append((source, dest))
        counts[(source, dest)] = []

    for time in sorted(times.keys()):
        for edge in counts:
            count = 0
            if len(counts[edge]) > 0:
                count = counts[edge][- 1]
            if edge in times[time]:
                count += 1
            counts[edge].append(count)

    step_times = pylab.arange(1, len(times) + 1, dtype=pylab.float64)
    pylab.figure()
    for edge in counts:
        counts[edge] /= step_times
        pylab.plot(step_times, counts[edge], label=edge)
    pylab.legend()
    pref = fname[:-4]
    pylab.savefig('%s.pdf' % pref, format='pdf')
    pylab.close()

def main():
    graph_names = argv[1:]
    for graph_name in graph_names:
        fname = 'critical_nodes_%s.csv' % graph_name
        critical_node_props(fname)

if __name__ == '__main__':
    main()
