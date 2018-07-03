import matplotlib as mpl
mpl.use('agg')
from matplotlib import animation
import pandas as pd
import networkx as nx
from sys import argv
from random import sample
import pylab
from networkx.algorithms import approximation
from collections import OrderedDict

DEFAULT_REPEATABILITY = 1
DEFAULT_LENGTH = 5 # cm

def bootstrap_clustering(G):
    deviations = []
    for sample_size in xrange(10, G.number_of_nodes(), 10):
        vals = []
        for i in xrange(10000):
            nbunch = sample(G.nodes(), sample_size)
            vals.append(nx.average_clustering(G.subgraph(nbunch)))
        print sample_size
        print pylab.std(vals, ddof=1)

def network_changes(network_file):
    df = pd.read_csv('mapping_network/csv/turtle_hill.csv', skipinitialspace=True)
    #G = nx.MultiGraph()
    G = nx.Graph()
    path_nodes = set()
    nodes_used = OrderedDict()
    for row in df.iterrows():
        row = row[1]
        nodes = row['Nodes']
        nodes = nodes.split('to')
        nodes = map(lambda x : x.strip(), nodes)
        assert len(nodes) >= 2
        for i in xrange(1, len(nodes)):
            n1, n2 = nodes[i - 1], nodes[i]
            n1 = n1.strip()
            n2 = n2.strip()
            r1 = row['Repeatability index from nest out']
            r2 = row['Repeatability index toward nest']
            repeatabilities = []
            for repeatability in (r1, r2):
                if not pd.isnull(repeatability):
                    repeatabilities.append(int(repeatability))
            repeatability = None
            if len(repeatabilities) > 0:
                repeatability = max(repeatabilities)
            else:
                repeatability = DEFAULT_REPEATABILITY
            G.add_edge(n1, n2)
            G[n1][n2]['repeatability'] = repeatability
        
        used_map = {'yes' : True, 'y' : True, 'no' : False, 'n' : False, 'No' : False}
        for key, item in row.iteritems():
            if 'used' in key:
                used = None
                if pd.isnull(item):
                    used = False
                else:
                    item = item.strip()
                    used = used_map[item]
                if used:
                    if key in nodes_used:
                        nodes_used[key].update([n1, n2])
                    else:
                        nodes_used[key] = set([n1, n2])

    nodes_used = nodes_used.items()

    contractions = set()
    for line in open('mapping_network/csv/turtle_hill_contract.csv'):
        line = line.strip('\n')
        line = line.split(' = ')
        line = sorted(line, key=lambda x : len(x))
        line = map(lambda x : x.strip(), line)

        n1 = line[0]
        for n2 in line[1:]:
            if (n1, n2) in contractions:
                continue
            G = nx.contracted_nodes(G, n1, n2)
            contractions.add((n1, n2))

    terminals = []
    for line in open('mapping_network/csv/turtle_hill_terminals.csv'):
        line = line.strip('\n')
        node, terminal = line.split('to')
        node = node.strip()
        terminal = terminal.strip()
        terminals.append(terminal)
        G.add_edge(node, terminal)
        G[node][terminal]['repeatability'] = DEFAULT_REPEATABILITY
        G[node][terminal]['length'] = DEFAULT_LENGTH
        
    print terminals
    
    S = nx.algorithms.approximation.steinertree.steiner_tree(G, terminals)
    
    graphscale = 1
    pos = nx.kamada_kawai_layout(G, scale = graphscale)
    labels = {}
    max_node = float("-inf")
    for u in G.nodes():
        if u.isdigit():
            labels[u] = u
            max_node = max(max_node, int(u))

    repeatability_color = {1 : 'k', 2 : 'b', 3 : 'r', 4 : 'g'}
    edgelist = []
    edge_color = []
    for u, v in G.edges():
        edgelist.append((u, v))
        repeatability = G[u][v]['repeatability']
        color = repeatability_color[repeatability]
        edge_color.append(color)
    
    next_node = int(max_node) + 1
    for u in G.nodes():
        if u not in labels:
            labels[u] = str(next_node)
            next_node += 1 

    fig = pylab.figure()
    def init():
        pylab.clf()

    def redraw(frame):
        print frame
        day, path_nodes = nodes_used[frame]

        nodelist = []
        node_color = []
        for u in G.nodes():
            nodelist.append(u)
            if S.has_node(u):
                node_color.append('m')
            elif u in path_nodes:
                node_color.append('y')
            else:
                node_color.append('r')

        nx.draw(G, pos=pos, with_labels=True, node_size=20, font_size=5,\
                labels=labels, node_color=node_color, nodelist=nodelist,\
                edgelist=edgelist, edge_color=edge_color)
        pylab.draw()
        pylab.savefig('mapping_network/figs/mapping_network%d.pdf' % frame, format='pdf')
        #pylab.close()
    
    ani = animation.FuncAnimation(fig, redraw, frames=len(nodes_used), interval=10000, init_func=init)
    mywriter = animation.AVConvWriter()
    ani.save('mapping_network/figs/mapping_network.mp4', writer=mywriter)
    pylab.close()

def main():
    network_changes('mapping_network/csv/turtle_hill.csv')

if __name__ == '__main__':
    main()
