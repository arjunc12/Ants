import matplotlib as mpl
mpl.use('agg')
from matplotlib import animation
import pandas as pd
import networkx as nx
from sys import argv
from random import sample
import pylab
from networkx.algorithms import approximation

def bootstrap_clustering(G):
    deviations = []
    for sample_size in xrange(10, G.number_of_nodes(), 10):
        vals = []
        for i in xrange(10000):
            nbunch = sample(G.nodes(), sample_size)
            vals.append(nx.average_clustering(G.subgraph(nbunch)))
        print sample_size
        print pylab.std(vals, ddof=1)

def main():
    df = pd.read_csv('mapping_network/csv/turtle_hill.csv', skipinitialspace=True)
    G = nx.MultiGraph()
    path_nodes = set()
    for row in df.iterrows():
        row = row[1]
        nodes = row['Nodes']
        n1, n2 = nodes.split('to')
        n1 = n1.strip()
        n2 = n2.strip()
        r1 = row['Repeatability index from nest out']
        r2 = row['Repeatability index toward nest']
        repeatability = int(max(r1, r2))
        for i in xrange(repeatability):
            G.add_edge(n1, n2)
        
        used = row['used 6-24-18']
        used_map = {'yes' : True, 'y' : True, 'no' : False, 'n' : False, 'No' : False}
        used = used_map[used]
        if used:
            path_nodes.update([n1, n2])

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

    graphs = [G, G]

    fig = pylab.figure()
    def init():
        pylab.clf()

    def redraw(frame):
        print frame
        G = nx.Graph(graphs[frame])
        terminals = ['nest 1', 'nest 2', 'nest 3']

        G.add_edges_from(zip(terminals, ['1', '18', '71']))
        S = nx.algorithms.approximation.steinertree.steiner_tree(G, terminals)
        
        graphscale = 1
        pos = nx.kamada_kawai_layout(G, scale = graphscale)
        labels = {}
        max_node = float("-inf")
        nodelist = []
        node_color = []
        for u in G.nodes():
            if u.isdigit():
                labels[u] = u
                max_node = max(max_node, int(u))

        next_node = int(max_node) + 1
        nodelist = []
        node_color = []
        for u in G.nodes():
            nodelist.append(u)
            if u not in labels:
                labels[u] = str(next_node)
                next_node += 1
            if S.has_node(u):
                node_color.append('m')
            elif u in path_nodes:
                node_color.append('y')
            else:
                node_color.append('r')

        nx.draw(G, pos=pos, with_labels=True, node_size=20, font_size=5, labels=labels, node_color=node_color, nodelist=nodelist)
        pylab.draw()
        #pylab.savefig('mapping_network/figs/mapping_network%d.pdf' % frame, format='pdf')
        #pylab.close()
    
    ani = animation.FuncAnimation(fig, redraw, frames=len(graphs), interval=100, init_func=init)
    ani.save('mapping_network/figs/mapping_network.mp4')
    pylab.close()


if __name__ == '__main__':
    main()
