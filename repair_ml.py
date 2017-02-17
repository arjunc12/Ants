import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from sys import argv
import pandas as pd
import numpy as np
import pylab
import argparse
import numpy.ma as ma
from choice_functions import *
import os

MIN_PHEROMONE = 0
MIN_DETECTABLE_PHEROMONE = 0
INIT_WEIGHT = 0

IND_PLOT = False
#THRESHOLD = 1
#EXPLORE_PROB = 0.1

def reset_graph(G):
    for u, v in G.edges_iter():
        G[u][v]['weight'] = MIN_PHEROMONE
        G[u][v]['units'] = []

def make_graph(sources, dests):
    assert len(sources) == len(dests)
    G = nx.Graph()
    for i in xrange(len(sources)):
        source = sources[i]
        dest = dests[i]
        G.add_edge(source, dest)
        G[source][dest]['weight'] = INIT_WEIGHT #MIN_PHEROMONE
        G[source][dest]['units'] = []
        
    return G

def check_graph(G, check_units=False):
    for u, v in G.edges_iter():
        weight = G[u][v]['weight']
        assert weight >= MIN_PHEROMONE
        if check_units:
            units = G[u][v]['units']
            if len(units) == 0:
                assert weight == MIN_PHEROMONE
            else:
                wt = 0
                for unit in G[u][v]['units']:
                    assert unit > MIN_PHEROMONE
                    wt += unit
                assert wt == weight

def edge_weight(G, u, v):
    units = G[u][v]['units']
    wt = MIN_PHEROMONE
    if len(units) > 0:
        wt = 0
        for unit in units:
            assert unit > MIN_PHEROMONE
            wt += unit
        assert wt > MIN_PHEROMONE
    assert wt >= MIN_PHEROMONE
    return wt

def decay_units(G, u, v, decay, seconds = 1):
    nonzero_units = []
    for unit in G[u][v]['units']:
        unit = max(unit - (decay * seconds), MIN_PHEROMONE)
        assert unit >= MIN_PHEROMONE
        if unit > MIN_PHEROMONE:
            nonzero_units.append(unit)
    G[u][v]['units'] = nonzero_units

def decay_graph_const(G, decay, seconds=1):
    wt = G[u][v]['weight']
    assert wt >= MIN_PHEROMONE
    x = max(MIN_PHEROMONE, wt - (decay * seconds))
    G[u][v]['weight'] = x
    
def decay_graph_linear(G, decay, seconds=1):
    for u, v in G.edges_iter():
        decay_units(G, u, v, decay, seconds)
        wt = edge_weight(G, u, v)
        assert wt >= MIN_PHEROMONE
        G[u][v]['weight'] = wt

def decay_graph_exp(G, decay, seconds=1):
    assert decay > 0
    assert decay < 1
    decay_prop = (1 - decay) ** seconds
    for u, v in G.edges_iter():
        wt = G[u][v]['weight']
        new_wt = wt * ((1 - decay) ** seconds)
        if new_wt == wt:
            new_wt = MIN_PHEROMONE
        x = max(MIN_PHEROMONE, new_wt)
        G[u][v]['weight'] = x

def get_decay_func(decay_type):
    if decay_type == 'const':
        return decay_graph_const
    elif decay_type == 'linear':
        return decay_graph_linear
    elif decay_type == 'exp':
        return decay_graph_exp
    else:
        raise ValueError("Invalid Decay Type")

def param_likelihood(choices, decay, explore, likelihood_func, decay_type, G=None):
    assert 0 < decay < 1
    df = pd.read_csv(choices, header=None, names=['source', 'dest', 'dt'], skipinitialspace=True)
    df['dt'] = pd.to_datetime(df['dt'])
    df.sort('dt', inplace=True)
    sources = list(df['source'])
    dests = list(df['dest'])
    dts = list(df['dt'])
    
    assert len(sources) == len(dests)
    
    if G == None:
        G = make_graph(sources, dests)
    else:
        reset_graph(G)
        
    decay_func = get_decay_func(decay_type)
    check_units = False
    if decay_type == 'linear':
        check_units = True
    
    log_likelihood = 0
    G[sources[1]][dests[1]]['weight'] += 1
    if decay_type == 'linear':
        G[sources[1]][dests[1]]['units'].append(1)
    G2 = G.copy()
    for i in xrange(1, len(sources)):
        check_graph(G, check_units)
        check_graph(G2, check_units)
        
        source = sources[i]
        dest = dests[i]
        
        log_likelihood += np.log(likelihood_func(G, source, dest, explore))
        if log_likelihood == float("-inf"):
            break
        
        curr = dts[i]
        prev = dts[i - 1]
        if curr != prev:
            diff = curr - prev
            seconds = diff.total_seconds()
            G = G2
            decay_func(G, decay, seconds)
            G2 = G.copy()
        G2[source][dest]['weight'] += 1
        if decay_type == 'linear':
            G2[source][dest]['units'].append(1)
               
    return log_likelihood, G

def max_likelihood_estimates(likelihoods, decays, explores):
    max_likelihood = float("-inf")
    max_values = []
    pos = 0
    for decay in decays:
        for explore in explores:
            i, j = pos / len(explores), pos % len(explores)
            pos += 1
            likelihood = likelihoods[i, j]
            if likelihood == float("-inf"):
                continue
            if likelihood > max_likelihood:
                max_values = [(explore, decay)]
                max_likelihood = likelihood
            elif max_likelihood == likelihood:
                max_values.append((explore, decay))
    return max_likelihood, max_values
    
def likelihood_matrix(sheet, explores, decays, likelihood_func, decay_type):
    likelihoods = pylab.zeros((len(decays), len(explores)))
    G = None
    choices = 'reformated_counts%s.csv' % sheet
    pos = 0

    for decay in decays:
        for explore in explores:
            i, j = pos / len(explores), pos % len(explores)
            likelihood, G = param_likelihood(choices, decay, explore, likelihood_func, decay_type, G)
            #likelihood = pos
            likelihoods[i, j] = likelihood
            pos += 1        
    return likelihoods

def get_likelihood_func(strategy):
    if strategy == 'uniform':
        return uniform_likelihood
    elif strategy == 'max':
        return max_edge_likelihood
    elif strategy == 'maxz':
        return maxz_edge_likelihood
    elif strategy == 'rank':
        return rank_likelihood
    else:
        raise ValueError('invalid strategy')

def make_title_str(max_likelihood, max_values):
    title_str = ['max likelihood %f at:' % max_likelihood]
    for explore, decay in max_values:
        title_str.append('(e=%0.2f, d=%0.2f)' % (explore, decay))
    title_str = '\n'.join(title_str)
    return title_str

def plot_likelihood_heat(likelihoods, max_likelihood, max_values, explores, decays, outname):
    likelihoods = ma.masked_invalid(likelihoods)
    title_str = make_title_str(max_likelihood, max_values)
    print title_str
    pylab.figure()
    hm = pylab.pcolormesh(likelihoods, cmap='nipy_spectral')
    curr_ax = pylab.gca()
    curr_ax.axis('tight')
    cb = pylab.colorbar(hm)
    cb.ax.set_ylabel('log-likelihood', fontsize=20)
    pylab.tick_params(which='both', bottom='off', top='off', left='off', right='off', \
    labeltop='off', labelbottom='off', labelleft='off', labelright='off')
    
    pylab.xlabel("explore probability (%0.2f - %0.2f)" % (min(explores), max(explores)), fontsize=20)
    pylab.ylabel("pheromone decay (%0.2f-%0.2f)" % (min(decays), max(decays)), fontsize=20)
    #pylab.title(title_str)
    pylab.savefig(outname + '.png', format="png", transparent=True, bbox_inches='tight')
    pylab.close()
    print 'convert %s.png %s.pdf' % (outname, outname)
    os.system('convert %s.png %s.pdf' % (outname, outname))

def ml_heat(label, sheets, strategies, decay_types, dmin=0.05, dmax=0.95, emin=0.05, \
            emax=0.95, dstep=0.05, estep=0.05, cumulative=False, out=False):
    decays = np.arange(dmin, dmax + dstep, dstep)
    explores = np.arange(emin, emax + estep, estep)
    hist_file = open('repair_ml_hist.csv', 'a')
    for strategy in strategies:
        print strategy
        likelihood_func = get_likelihood_func(strategy)
        for decay_type in decay_types:
            print decay_type
            out_str = 'repair_ml_%s_%s' % (strategy, decay_type)
            cumulative_likelihoods = None
            total_lines = 0
            for sheet in sheets:
                print sheet
                num_lines = sum(1 for line in open('reformated_counts%s.csv' % sheet))
                total_lines += num_lines
                likelihoods = likelihood_matrix(sheet, explores, decays, likelihood_func,\
                                                decay_type)
                max_likelihood, max_values = max_likelihood_estimates(likelihoods, decays, explores)
                outname = '%s_%s' % (out_str, sheet)
                
                if cumulative:
                    if not isinstance(cumulative_likelihoods, np.ndarray):
                        cumulative_likelihoods = np.copy(likelihoods)
                    else:
                        cumulative_likelihoods += likelihoods
                        
                if IND_PLOT:
                    plot_likelihood_heat(likelihoods, max_likelihood, max_values, explores, \
                                          decays, outname)
                if out:
                    for explore, decay in max_values:
                        hist_file.write('%0.2f, %0.2f, %s, %s, %s, %d\n' % \
                                        (explore, decay, strategy, decay_type, label, num_lines))
            
            if cumulative:
                cumulative_likelihoods
                outname = 'cumulative_%s_%s' % (out_str, label)
                max_likelihood, max_values = max_likelihood_estimates(cumulative_likelihoods,\
                                                              decays, explores)
                plot_likelihood_heat(cumulative_likelihoods, max_likelihood, max_values, \
                                explores, decays, outname)
                print "plotted"
    hist_file.close()
                
    
if __name__ == '__main__':
    strategy_choices = ['uniform', 'max', 'maxz', 'rank']
    decay_choices = ['linear', 'const', 'exp']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('sheets', nargs='+')
    parser.add_argument('-s', '--strategies', action='store', nargs='+', \
                        choices=strategy_choices, required=True)
    parser.add_argument('-d', '--decay_types', nargs='+', choices=decay_choices, required=True)
    parser.add_argument('-c', '--cumulative', action='store_true')
    parser.add_argument('-dmin', type=float, default=0.05)
    parser.add_argument('-dmax', type=float, default=0.95)
    parser.add_argument('-emin', type=float, default=0.05)
    parser.add_argument('-emax', type=float, default=0.95)
    parser.add_argument('-dstep', type=float, default=0.05)
    parser.add_argument('-estep', type=float, default=0.05)
    parser.add_argument('-o', '--out', action='store_true')
    
    args = parser.parse_args()
    label = args.label
    sheets = args.sheets
    strategies = args.strategies
    decay_types = args.decay_types
    dmin, dmax, emin, emax = args.dmin, args.dmax, args.emin, args.emax
    dstep, estep = args.dstep, args.estep
    cumulative = args.cumulative
    out = args.out
    
    ml_heat(label, sheets, strategies, decay_types, dmin, dmax, emin, emax, dstep, estep, cumulative, out)
