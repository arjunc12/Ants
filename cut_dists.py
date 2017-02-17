from repair_ml import make_graph
import networkx as nx
from sys import argv
import pylab
import pandas as pd

def get_dists(counts_file, cut_node):
    dists = []
    converters = {'source' : str, 'dest' : str}
    choices = pd.read_csv(counts_file, header=None, names=['source', 'dest', 'dt'], skipinitialspace=True, \
                          converters=converters)
    #df['dt'] = pd.to_datetime(df['dt'])
    #df.sort('dt', inplace=True)
    sources = list(choices['source'])
    dests = list(choices['dest'])
    #dts = list(df['dt'])
    
    assert len(sources) == len(dests)
    
    G = make_graph(sources, dests)
    print G.nodes()
    print G.edges()
    
    for i in xrange(len(sources)):
        source, dest = sources[i], dests[i]
        if not nx.has_path(G, source, cut_node) and not nx.has_path(G, dest, cut_node):
            continue
        dist1 = nx.shortest_path_length(G, source, cut_node)
        dist2 = nx.shortest_path_length(G, dest, cut_node)
        dists.append(min(dist1, dist2))
        
    return dists

def dists_hist(cuts):
    dists = []
    for counts_file, cut_node in cuts:
        dists += get_dists(counts_file, cut_node)
    pylab.hist(dists)
    print "show"
    pylab.show()
    
def main():
    args = argv[1:]
    assert len(args) % 2 == 0
    cuts = []
    for i in xrange(0, len(args), 2):
        cuts.append(('reformated_counts' + args[i] + '.csv', args[i + 1]))
    dists_hist(cuts)
    
if __name__ == '__main__':
    main()