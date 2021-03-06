import matplotlib as mpl
mpl.use('agg')
import pylab
import pandas as pd
import networkx as nx
from collections import defaultdict
import seaborn as sns
import argparse
import os
from datetime import timedelta

PATH1 = set(['5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',\
         '19', '20', '21', '22', '22A', '23'])

def get_df():
    df = pd.read_csv('repeatability/csv/repeatability1.csv')
    df['Different plant'] = df['Different plant'].apply(lambda x: x.strip())
    return df

def length_distribution(df=None):
    if df is None:
        df = get_df()
    df2 = df.copy()
    df2['Distance in cm'] = df2['Distance in mm'] * 0.1
    #df2 = df2[df2['Distance in mm'] < 80]
    print "mean", pylab.mean(df2['Distance in cm'])
    print "std", pylab.std(df2['Distance in cm'], ddof=1)
    for name, group in df2.groupby('Different plant'):
        print name
        print pylab.mean(group['Distance in cm'])
    pylab.figure()
    pylab.hist(df2['Distance in cm'])
    pylab.xlabel('Distance in cm')
    pylab.ylabel('count')
    pylab.savefig('repeatability/figs/distances.pdf', format='pdf')
    pylab.close()

def difficulty_distributions(df=None):
    if df is None:
        df = get_df()
    distributions = {}
    for name, group in df.groupby('Different plant'):
        distribution = defaultdict(int)
        observations = 0
        for difficulty in group['Junction difficulty']:
            distribution[difficulty] += 1
            observations += 1

        for difficulty in distribution:
            distribution[difficulty] /= float(observations)
            
        distributions[name] = distribution
        
    return distributions

def difficulty_barplot(df):
    x = []
    y = []
    hue = []
    group_sums = {}
    for name, group in df.groupby('Different plant'):
        group_sums[name] = float(group.size)
    for name, group in df.groupby(['Junction difficulty', 'Different plant']):
        difficulty, plant = name
        x.append(difficulty)
        y.append(group.size / group_sums[plant])
        hue.append(plant + ' plant')

    pylab.figure()
    sns.barplot(x=x, y=y, hue=hue, ci=None)
    pylab.xlabel('edge difficulty')
    pylab.ylabel('proportion')
    pylab.savefig('repeatability/figs/repeatability.pdf', format='pdf')
    pylab.close()

def difficulty_hist(df):
    difficulties = []
    weights = []
    labels = []
    pylab.figure()
    for name, group in df.groupby('Different plant'):
        difficulty = group['Junction difficulty']
        print name
        print pylab.mean(difficulty), "pm", pylab.std(difficulty, ddof=1)
        weight = pylab.ones_like(difficulty) / float(len(difficulty))
        difficulties.append(difficulty)
        weights.append(weight)
        labels.append(name + ' plant')
    pylab.hist(difficulties, bins=[1, 2, 3, 4, 5], label=labels, weights=weights)
    pylab.xlabel('edge difficulty')
    pylab.ylabel('frequency')
    pylab.legend()
    pylab.savefig('repeatability/figs/repeatability.pdf', format='pdf')
    pylab.close()

def read_plants():
    plants = {}
    with open('repeatability/csv/plants.csv') as f:
        i = 1
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            for node in line.split(','):
                plants[node] = i
            i += 1
    return plants

def make_graph(df):
    edges = df['Nodes']
    plants = df['Different plant']
    lengths = df['Distance in mm']
    assert len(edges) == len(plants)
    G = nx.Graph()
    for i, edge in enumerate(sorted(edges)):
        delim = None
        delim_options = ['-', 'to']
        for option in delim_options:
            if option in edge:
                delim = option
                break
        u, v = edge.split(delim)
        u, v = u.strip(), v.strip()
        G.add_edge(u, v)
        G[u][v]['plant'] = plants[i]
        G[u][v]['length'] = lengths[i]
        
    return G

def draw_network(df):
    G = make_graph(df)
    plants = read_plants()
    print plants
    queue = plants.keys()
    visited = set()
    while len(queue) > 0:
        curr = queue.pop(0)
        assert curr in plants
        visited.add(curr)
        for neighbor in G.neighbors(curr):
            if (G[curr][neighbor]['plant'] == 'same') and (neighbor not in visited)\
                                                      and (neighbor not in plants):
                plants[neighbor] = plants[curr]
                queue.append(neighbor)
                
    labels = {}
    next_label = max(plants.values()) + 1
    node_color = []
    nodelist = sorted(G.nodes())
    for u in nodelist:
        if u in PATH1:
            node_color.append('r')
        else:
            node_color.append('b')
        if u in plants:
            labels[u] = plants[u]
        else:
            print "new label", u, next_label, G.neighbors(u)
            labels[u] = next_label
            next_label += 1
                
 
    edge_color = []
    edgelist = sorted(G.edges())
    for u, v in edgelist:
        if G[u][v]['plant'] == 'same':
            edge_color.append('k')
        else:
            edge_color.append('b')

    pylab.figure()
    nx.draw(G, nodelist=nodelist, node_color=node_color, edgelist=edgelist,\
            edge_color=edge_color, with_labels=True, labels=labels)
    pylab.draw()
    pylab.savefig('repeatability/figs/network.pdf', format='pdf')
    pylab.close()
    
def graph_stats(df):
    G = make_graph(df)
    print G.number_of_nodes(), "nodes"
    print G.number_of_edges(), "edges"
    degrees = []
    for u in G.nodes_iter():
        degrees.append(G.degree(u))
    bins = range(max(degrees) + 1)
    weights = pylab.ones_like(degrees) / float(len(degrees))
    pylab.figure()
    pylab.hist(degrees, bins=bins, weights=weights)
    pylab.savefig('repeatability/figs/degree_dist.pdf', format='pdf')
    
    print "transitivity", nx.transitivity(G)
    print "average clustering", nx.average_clustering(G)

    shortest_path_lengths = nx.shortest_path_length(G, weight=None)
    visited = set()
    path_lens = []
    for source in shortest_path_lengths:
        for target in shortest_path_lengths[source]:
            if source == target:
                continue
            if tuple(sorted((source, target))) in visited:
                continue
            path_lens.append(shortest_path_lengths[source][target])
            visited.add(tuple(sorted((source, target))))
    print "mean shortest path len", pylab.mean(path_lens)
    print "diameter", max(path_lens)
    
    weights = pylab.ones_like(path_lens) / float(len(path_lens))
    pylab.figure()
    pylab.hist(path_lens, weights=weights)
    #pylab.xlabel('shortest path length (mm)')
    pylab.savefig('repeatability/figs/path_lengths.pdf', format='pdf')
    
    pylab.close()

def write_counts():
    counts_dir = 'repeatability/csv/counts'
    reformatted_counts_dir = 'repeatability/csv/reformatted_counts'
    
    time_key = 'Time (min after start)'
    
    all_counts_file = open('repeatability/csv/repeatability_counts.csv', 'w')
    all_counts_file.write('%s, %s, %s\n' % ('observation', 'repeatability', 'result'))
    
    for fname in os.listdir(counts_dir):
        df = pd.read_csv('%s/%s' % (counts_dir, fname), skipinitialspace=True)
        
        reformatted_counts_file = open('%s/%s' % (reformatted_counts_dir, fname), 'w')
        reformatted_counts_file.write('%s, %s, %s\n' % ('time', 'source', 'dest'))
        
        for row in df.iterrows():
            row = row[1]
            
            minutes = int(row[time_key])
            minutes = timedelta(minutes=minutes)
            
            for key, val in row.iteritems():
                if key == time_key:
                    continue
                elif pd.isnull(val):
                    continue
                
                source, dest = key.split('-')
                
                for obs in val.split(','):
                    obs = obs.strip()
                    
                    result = obs[-1]
                    
                    times = None
                    if len(result) == 1:
                        times = 1
                    else:
                        times = int(obs[:-1])
                    
                    for i in xrange(times):
                        reformatted_counts_file.write('%s, %s, %s\n' % (minutes, source, dest))

                        if source == 'B' and dest.isdigit():
                            all_counts_file.write('%s, %d, %s\n' % (fname[:-4], int(dest), result))

        reformatted_counts_file.close()
    all_counts_file.close()
                       
def counts_stats():
    df = pd.read_csv('repeatability/csv/repeatability_counts.csv', skipinitialspace=True)
    for repeatability, group in df.groupby('repeatability'):
        print repeatability
        result = group['result']
        total = result[result == 'T']
        print len(total), len(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lengths', action='store_true')
    parser.add_argument('-d', '--difficulties', action='store_true')
    parser.add_argument('-gs', '--graph_stats', action='store_true')
    parser.add_argument('-wc', '--write_counts', action='store_true')
    parser.add_argument('-cs', '--counts_stats', action='store_true')
    
    args = parser.parse_args()
    
    repeatability_df = get_df()
    
    if args.lengths:
        length_distribution(repeatability_df)
    if args.difficulties:
        difficulty_hist(df)
        difficulty_barplot(df)
        draw_network(df)
        print difficulty_distributions()
    
    if args.graph_stats:
        graph_stats(repeatability_df)
        
    if args.write_counts:
        write_counts()

    if args.counts_stats:
        counts_stats()

if __name__ == '__main__':
    main()
 
