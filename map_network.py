import matplotlib as mpl
mpl.use('agg')
from matplotlib import animation
import pandas as pd
import networkx as nx
from sys import argv
from random import sample
import pylab
from networkx.algorithms import approximation
from collections import OrderedDict, defaultdict
from itertools import combinations
import os

DEFAULT_REPEATABILITY = 1
DEFAULT_LENGTH = 5 # cm

DATA_DIR = 'mapping_network/csv'

def bootstrap_clustering(G):
    deviations = []
    for sample_size in xrange(10, G.number_of_nodes(), 10):
        vals = []
        for i in xrange(10000):
            nbunch = sample(G.nodes(), sample_size)
            vals.append(nx.average_clustering(G.subgraph(nbunch)))
        print sample_size
        print pylab.std(vals, ddof=1)

def similar_nodes(G):
    for u, v in combinations(list(G.nodes()), 2):
        n1 = sorted(G.neighbors(u))
        n2 = sorted(G.neighbors(v))
        if n1 == n2:
            print u, v, n1

def read_network(network):
    df = pd.read_csv('%s/%s/network.csv' % (DATA_DIR, network), skipinitialspace=True)
    G = nx.Graph()
    G.graph['name'] = network
    for row in df.iterrows():
        row = row[1]
        nodes = row['Nodes']
        if pd.isnull(nodes):
            continue
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
         
    terminals = []
    food_nodes = set()
    nests = set()
    
    for line in open('%s/%s/terminals.csv' % (DATA_DIR, network)):
        line = line.strip('\n')
        node, terminal = line.split('to')
        node = node.strip()
        terminal = terminal.strip()
        terminals.append(terminal)
        G.add_edge(node, terminal)
        G[node][terminal]['repeatability'] = DEFAULT_REPEATABILITY
        G[node][terminal]['length'] = DEFAULT_LENGTH

        if 'food' in terminal:
            food_nodes.add(terminal)
        elif 'nest' in terminal:
            nests.add(terminal)
    
    G.graph['nests'] = list(nests)
    G.graph['food nodes'] = list(food_nodes)
    
    contractions = nx.Graph()
    for line in open('%s/%s/contract.csv' % (DATA_DIR, network)):
        line = line.strip('\n')
        line = line.split('=')
        line = map(lambda x : x.strip(), line)

        assert len(line) >= 2
        
        for i in xrange(1, len(line)):
            contractions.add_edge(line[i - 1], line[i])
            
    for component in nx.connected_components(contractions):
        component = list(component)
        assert len(component) >= 2
        component = sorted(component, key = lambda x : len(x))
        n1 = component[0]
        assert G.has_node(n1)
        for i in xrange(1, len(component)):
            n2 = component[i]
            assert G.has_node(n2)
            G = nx.contracted_nodes(G, n1, n2)

    return G

def network_changes(network):
    df = pd.read_csv('%s/%s/network.csv' % (DATA_DIR, network), skipinitialspace=True)
    G = nx.Graph()
    path_nodes = set()
    nodes_used = OrderedDict()
    edge_order = []
    for row in df.iterrows():
        row = row[1]
        nodes = row['Nodes']
        if pd.isnull(nodes):
            continue
        nodes = nodes.split('to')
        nodes = map(lambda x : x.strip(), nodes)
        assert len(nodes) >= 2
        edge_tuple = []
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
            if len(edge_tuple) == 0:
                edge_tuple += [n1, n2]
            else:
                edge_tuple.append(n2)
            G[n1][n2]['repeatability'] = repeatability
        
        edge_order.append(tuple(edge_tuple))
        
        used_map = {'yes' : True, 'y' : True, 'Yes' : True, 'no' : False, 'n' : False, 'No' : False, 'N' : False}
        for key, item in row.iteritems():
            if 'used' in key.lower():
                used = None
                if pd.isnull(item):
                    used = False
                else:
                    item = item.strip()
                    used = used_map[item]

                if key not in nodes_used:
                    nodes_used[key] = set()
                
                if used:
                    nodes_used[key].update([n1, n2])
        

    print G.number_of_nodes(), "nodes before contraction"

    contractions = nx.Graph()
    for line in open('%s/%s/contract.csv' % (DATA_DIR, network)):
        line = line.strip('\n')
        line = line.split('=')
        line = map(lambda x : x.strip(), line)

        assert len(line) >= 2
        
        for i in xrange(1, len(line)):
            contractions.add_edge(line[i - 1], line[i])
            
    nodes_contracted = 0
    contraction_parents = {}
    for component in nx.connected_components(contractions):
        component = list(component)
        assert len(component) >= 2
        component = sorted(component, key = lambda x : len(x))
        n1 = component[0]
        assert G.has_node(n1)
        for i in xrange(1, len(component)):
            n2 = component[i]
            assert G.has_node(n2)
            G = nx.contracted_nodes(G, n1, n2)
            nodes_contracted += 1
            contraction_parents[n2] = n1

            for day, used in nodes_used.iteritems():
                if n2 in used:
                    nodes_used[day].add(n1)

    print nodes_contracted, "nodes contracted"
    print G.number_of_nodes(), "nodes after contraction"
    
    nodes_used = nodes_used.items()
            
    terminals = []
    for line in open('%s/%s/terminals.csv' % (DATA_DIR, network)):
        line = line.strip('\n')
        node, terminal = line.split('to')
        node = node.strip()
        terminal = terminal.strip()
        terminals.append(terminal)
        G.add_edge(node, terminal)
        G[node][terminal]['repeatability'] = DEFAULT_REPEATABILITY
        G[node][terminal]['length'] = DEFAULT_LENGTH

    S = nx.algorithms.approximation.steinertree.steiner_tree(G, terminals, weight='repeatability')
    
    graphscale = 1
    pos = nx.kamada_kawai_layout(G, scale = graphscale)
    labels = {}
    max_node = float("-inf")
    for u in G.nodes():
        if u.isdigit():
            labels[u] = u
            max_node = max(max_node, int(u))

    def get_label(u):
        if u in labels:
            return labels[u]
        else:
            assert u in contraction_parents
            parent = contraction_parents[u]
            assert parent in labels
            return labels[parent]

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

    with open('%s/%s/node_labels.csv' % (DATA_DIR, network), 'w') as f:
        for node, label in sorted(list(labels.iteritems()), key=lambda (x, y) : int(y)):
            f.write('%s, %s\n' % (node, label))

    with open('%s/%s/edge_labels.csv' % (DATA_DIR, network), 'w') as f:
        for edge in edge_order:
            edge_str = []
            for u in edge:
                label = get_label(u)
                edge_str.append(str(label))
            edge_str = ' to '.join(edge_str)
            f.write(edge_str + '\n')

    fig = pylab.figure()
    def init():
        pylab.text(0, 1.2, 'network structure', fontsize=15,\
                   verticalalignment='center', horizontalalignment='center',)
        nodelist = []
        node_color = []
        node_size = []
        for u in G.nodes():
            nodelist.append(u)
            if u in terminals:
                node_size.append(200)
            else:
                node_size.append(100)

            if S.has_node(u):
                node_color.append('m')
            else:
                node_color.append('k')

        nx.draw(G, pos=pos, with_labels=True, node_size=node_size, font_size=5,\
                labels=labels, node_color=node_color, nodelist=nodelist,\
                edgelist=edgelist, edge_color=edge_color, font_color='w')
        pylab.draw()
        pylab.savefig('mapping_network/figs/%s_init.pdf' % network, format='pdf')

    def redraw(frame):
        print frame
        if frame == 0:
            init()
        else:
            
            pylab.clf()
            
            day, path_nodes = nodes_used[frame - 1]

            day = day.strip('used ')
            pylab.text(0, 1.2, day, fontsize=15,\
                       verticalalignment='center', horizontalalignment='center')

            nodelist = []
            node_color = []
            node_size = []
            for u in G.nodes():
                nodelist.append(u)
                if u in terminals:
                    node_size.append(200)
                else:
                    node_size.append(100)

                if u in path_nodes:
                    if S.has_node(u):
                        node_color.append('b')
                    else:
                        node_color.append('r')
                else:
                    if S.has_node(u):
                        node_color.append('m')
                    else:
                        node_color.append('k')

            nx.draw(G, pos=pos, with_labels=True, node_size=node_size, font_size=5,\
                    labels=labels, node_color=node_color, nodelist=nodelist,\
                    edgelist=edgelist, edge_color=edge_color, font_color='w')
            pylab.draw()
            pylab.savefig('mapping_network/figs/%s%d.pdf' % (network, frame), format='pdf')
            #pylab.close()
    
    ani = animation.FuncAnimation(fig, redraw, frames=len(nodes_used) + 1, interval=5000, init_func=init)
    #mywriter = animation.AVConvWriter()
    ani.save('mapping_network/figs/%s.mp4' % network, writer='avconv', dpi=300)
    pylab.close()
    
    print "similar nodes"
    similar_nodes(G)

def main():
    for network in os.listdir(DATA_DIR):
        print network
        network_changes(network)

if __name__ == '__main__':
    main()
