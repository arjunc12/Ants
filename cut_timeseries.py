import pandas as pd
from datetime import timedelta
from sys import argv
import networkx as nx
import argparse
from choice_functions import *
from decay_functions import *
from repair_ml import reset_graph, make_graph
import pylab

def estimate_explore(df, start=0, G=None, strategy='rank', decay_type='exp', decay=0.01, ghost=False):
    df.sort('dt', inplace=True) 
    sources = list(df['source'])
    dests = list(df['dest'])
    dts = list(df['dt'])
    
    assert len(sources) == len(dests)
    assert start <= len(sources)
    
    if G == None:
        G = make_graph(sources, dests, ghost)
    else:
        reset_graph(G)
        
    steps = 0
    explore_steps = 0
    
    G2 = G.copy()
    
    G[sources[0]][dests[0]]['weight'] += 1
    if decay_type == 'linear':
        G[sources[0]][dests[0]]['units'].append(1)
        
    for i in xrange(1, len(sources)):
        
        curr = dts[i]
        prev = curr
        if i > 0:
            prev = dts[i - 1]
        
        source = sources[i]
        dest = dests[i]
        
        if i <= start:
            explore_step = is_explore(G, source, dest, strategy)
            if explore_step != None:
                steps += 1
                if explore_step == True:
                    explore_steps += 1
        
        if curr != prev:
            diff = curr - prev
            seconds = diff.total_seconds()
            G = G2
            decay_graph(G, decay_type, decay, seconds)
            G2 = G.copy()
        G2[source][dest]['weight'] += 1
            
        #print curr, get_weights(G)
            
    return explore_steps, steps

def time_series_explore(strategies, decay_types, sheets, decay=0.01, window=-1, ghost=False):
    for strategy in strategies:
        print strategy
        for decay_type in decay_types:
            print decay_type
            for sheet in sheets:
                print sheet
                choices = 'reformated_counts%s.csv' % sheet
                choices = pd.read_csv(choices, header=None, names=['source', 'dest', 'dt'], skipinitialspace=True)
                choices['dt'] = pd.to_datetime(choices['dt'])
                choices.sort('dt', inplace=True)
                delta = None
                if window != -1:
                    assert window > 0
                    delta = timedelta(minutes=window)
                timestamps = list(choices['dt'])
                explore_probs = []
                for i in xrange(len(timestamps)):
                    if i > 0 and timestamps[i] == timestamps[i - 1]:
                        continue
                    '''
                    upper_lim = timestamps[len(timestamps) - 1]
                    if delta != None:
                        upper_lim = timestamp + delta
                    df = choices[(choices['dt'] >= timestamp) & (choices['dt'] <= upper_lim)]
                    '''
                    #if len(df.index) == 0:
                    #    continue
                    explore_steps, steps = estimate_explore(choices, i, None, strategy, decay_type, decay, ghost)
                    if steps > 0:
                        explore_prob = float(explore_steps) / steps
                        explore_probs.append(explore_prob)
                pylab.plot(range(len(explore_probs)), explore_probs)
            print "show"
            pylab.show()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sheets', nargs='+')
    parser.add_argument('-s', '--strategies', choices=STRATEGY_CHOICES, nargs='+', required=True)
    parser.add_argument('-dt', '--decay_types', choices=DECAY_CHOICES, nargs='+', required=True)
    parser.add_argument('-d', '--decay', type=float, default=0.01)
    parser.add_argument('-w', '--window', type=int, default=-1)
    parser.add_argument('-g', '--ghost', action='store_true')
    
    args = parser.parse_args()
    sheets = args.sheets
    strategies = args.strategies
    decay_types = args.decay_types
    decay = args.decay
    window = args.window
    ghost = args.ghost
    
    time_series_explore(strategies, decay_types, sheets, decay, window, ghost)
    
    
if __name__ == '__main__':
    main()