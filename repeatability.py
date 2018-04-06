import matplotlib as mpl
mpl.use('agg')
import pylab
import pandas as pd
import networkx as nx
from collections import defaultdict
import seaborn as sns

def get_df():
    df = pd.read_csv('repeatability/csv/repeatability1.csv')
    df['Different plant'] = df['Different plant'].apply(lambda x: x.strip())
    return df

def difficulty_distributions():
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

def draw_network(df):
    edges = df['Nodes']
    plants = df['Different plant']
    assert len(edges) == len(plants)
    G = nx.Graph()
    for i, edge in enumerate(sorted(edges)):
        delim = None
        delim_options = ['-', 'to']
        for option in delim_options:
            if option in edge:
                delim = option
                break
        print edge
        print delim
        u, v = edge.split(delim)
        u, v = u.strip(), v.strip()
        G.add_edge(u, v)
        G[u][v]['plant'] = plants[i]
 
    edge_color = []
    edgelist = sorted(G.edges())
    for u, v in edgelist:
        if G[u][v]['plant'] == 'same':
            edge_color.append('k')
        else:
            edge_color.append('b')

    pylab.figure()
    nx.draw(G, edgelist=edgelist, edge_color=edge_color)
    pylab.draw()
    pylab.savefig('repeatability/figs/network.pdf', format='pdf')
    pylab.close()

def main():
    df = get_df()
    #difficulty_hist(df)
    #difficulty_barplot(df)
    draw_network(df)
    #print difficulty_distributions()

if __name__ == '__main__':
    main()
