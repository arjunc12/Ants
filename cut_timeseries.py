import pandas as pd
from datetime import timedelta
from sys import argv
import networkx as nx

MIN_PHEROMONE = 0
INIT_WEIGHT = 0

def reset_graph(G):
    for u, v in G.edges_iter():
        G[u][v]['weight'] = INIT_WEIGHT

def make_graph(sources, dests):
    assert len(sources) == len(dests)
    G = nx.Graph()
    for i in xrange(len(sources)):
        source = sources[i]
        dest = dests[i]
        G.add_edge(source, dest)
        G[source][dest]['weight'] = INIT_WEIGHT #MIN_PHEROMONE
    return G

def decay_graph(G, decay, seconds=1):
    for u, v in G.edges_iter():
        wt = G[u][v]['weight']
        assert wt >= MIN_PHEROMONE
        x = max(MIN_PHEROMONE, wt - (decay * seconds))
        assert wt >= MIN_PHEROMONE
        G[u][v]['weight'] = x

def get_weights(G):
    weights = []
    for u, v in G.edges():
        weights.append(G[u][v]['weight'])
    return weights

def estimate_explore(df, decay, G=None):
    #choices.sort('dt', inplace=True) 
    sources = list(df['source'])
    dests = list(df['dest'])
    dts = list(df['dt'])
    
    assert len(sources) == len(dests)
    
    if G == None:
        G = make_graph(sources, dests)
    else:
        reset_graph(G)
        
    steps = 0
    explore_steps = 0
    
    G[sources[1]][dests[1]]['weight'] += 1
    G2 = G.copy()
    for i in xrange(1, len(sources)):
        
        curr = dts[i]
        prev = dts[i - 1]
        
        source = sources[i]
        dest = dests[i]
        
        steps += 1
        if G[source][dest]['weight'] == MIN_PHEROMONE:
            explore_steps += 1
        
        if curr != prev:
            diff = curr - prev
            seconds = diff.total_seconds()
            G = G2
            decay_graph(G, decay, seconds)
            G2 = G.copy()
        G2[source][dest]['weight'] += 1
            
        #print curr, get_weights(G)
            
    return float(explore_steps) / steps

def time_series_explore(choices, decay, window):
    choices.sort('dt', inplace=True)
    delta = timedelta(minutes=window)
    timestamps = choices['dt']
    G = make_graph(choices['source'], choices['dest'])
    for i, timestamp in enumerate(timestamps):
        if i > 0 and timestamp == timestamps[i - 1]:
            continue
        upper_lim = timestamp + delta
        df = choices[(choices['dt'] >= timestamp) & (choices['dt'] <= upper_lim)]
        if len(df.index) == 0:
            continue
        explore_prob = estimate_explore(df, decay, G)
        print explore_prob
    
if __name__ == '__main__':
    cutdata = argv[1]
    df = pd.read_csv(cutdata, header=None, names=['source', 'dest', 'dt'])
    df['dt'] = pd.to_datetime(df['dt'])
    
    time_series_explore(df, 0.1, 3)