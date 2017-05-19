import matplotlib as mpl
mpl.use('agg')
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
from decay_functions import *
import os

MIN_PHEROMONE = 0
MIN_DETECTABLE_PHEROMONE = 0
INIT_WEIGHT = 0

IND_PLOT = False

NOCUT = ['6', '7a', '7c', '8a', '8b', '9b', '14','15', '16', '17', '21a', '21b', '23a', '23d']
CUT = ['7b', '9a', '9c', '9d', '10', '11', '12', '13', '18', '19', '20', '21c', '22', '23b', '23c', '23e', '23f']
ALL = NOCUT + CUT

NOCUT2 = ['6', '7a', '7c', '9b', '14','15', '16', '21a', '21b', '23a', '23d']
ALL2 = NOCUT2 + CUT

DATASETS_DIR = 'datasets/reformated_csv'
OUT_DIR = 'ml_plots'

def reset_graph(G, init_weight=INIT_WEIGHT):
    for u, v in G.edges_iter():
        G[u][v]['weight'] = init_weight
        G[u][v]['units'] = [init_weight]

def make_graph(sources, dests, ghost=False, init_weight=INIT_WEIGHT):
    assert len(sources) == len(dests)
    G = nx.Graph()
    for i in xrange(len(sources)):
        source = sources[i]
        dest = dests[i]
        G.add_edge(source, dest)
        G[source][dest]['weight'] = init_weight
        G[source][dest]['units'] = [init_weight]
    
    if ghost:    
        for u in G.nodes():
            if len(G.neighbors(u)) == 1:
                v = 'canopy_%s' % u
                G.add_edge(u, v)
                G[u][v]['weight'] = init_weight
                G[u][v]['units'] = [init_weight]
    
    #print G.nodes()
    #print G.edges()
        
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

def param_likelihood(choices, decay, explore, likelihood_func, decay_type, G=None, ghost=False):
    assert 0 <= decay <= 1
    assert 0 <= explore <= 1
    df = pd.read_csv(choices, header=None, names=['source', 'dest', 'dt'], skipinitialspace=True)
    df['dt'] = pd.to_datetime(df['dt'])
    df.sort('dt', inplace=True)
    sources = list(df['source'])
    dests = list(df['dest'])
    dts = list(df['dt'])
    
    assert len(sources) == len(dests)
    
    if G == None:
        G = make_graph(sources, dests, ghost)
    else:
        reset_graph(G)
        
    decay_func = get_decay_func_graph(decay_type)
    check_units = False
    if decay_type == 'linear':
        check_units = True
    
    log_likelihood = 0    
    G[sources[0]][dests[0]]['weight'] += 1
    if decay_type == 'linear':
        G[sources[0]][dests[0]]['units'].append(1)
    G2 = G.copy()
    
    max_degree = 0
    for u in G.nodes():
        max_degree = max(max_degree, len(G.neighbors(u)))
    
    for i in xrange(1, len(sources)):
        #check_graph(G, check_units)
        #check_graph(G2, check_units)
        
        source = sources[i]
        dest = dests[i]
        
        '''
        canopy_neighbor = 'canopy_%s' % source
        if canopy_neighbor in G.neighbors(source):
            G[source][canopy_neighbor]['weight'] += 1
            if decay_type == 'linear':
                G[source][canopy_neighbor % source]['units'].append(1)
        '''
        log_likelihood += np.log(likelihood_func(G, source, dest, explore))
        if log_likelihood == float("-inf"):
            break
        
        curr = dts[i]
        prev = curr
        if i > 0:
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
    
def likelihood_matrix(sheet, explores, decays, likelihood_func, decay_type, ghost):
    likelihoods = pylab.zeros((len(decays), len(explores)))
    G = None
    choices = '%s/reformated_counts%s.csv' % (DATASETS_DIR, sheet)
    pos = 0

    for decay in decays:
        for explore in explores:
            i, j = pos / len(explores), pos % len(explores)
            likelihood, G = param_likelihood(choices, decay, explore, likelihood_func, decay_type, G, ghost)
            #likelihood = pos
            likelihoods[i, j] = likelihood
            pos += 1        
    return likelihoods

def make_title_str(max_likelihood, max_values):
    title_str = ['max likelihood %f at:' % max_likelihood]
    for explore, decay in max_values:
        title_str.append('(e=%0.2f, d=%0.2f)' % (explore, decay))
    title_str = '\n'.join(title_str)
    return title_str

def plot_likelihood_heat(likelihoods, max_likelihood, max_values, explores,\
                         decays, outname):
    likelihoods = ma.masked_invalid(likelihoods)
    title_str = make_title_str(max_likelihood, max_values)
    print title_str
    pylab.figure()
    hm = pylab.pcolormesh(likelihoods, cmap='nipy_spectral')
    curr_ax = pylab.gca()
    curr_ax.axis('tight')
    curr_ax.set_aspect('equal')
    cb = pylab.colorbar(hm)
    cb.ax.set_ylabel('log-likelihood', fontsize=20)
    pylab.tick_params(which='both', bottom='off', top='off', left='off', right='off', \
    labeltop='off', labelbottom='off', labelleft='off', labelright='off')
    
    pylab.xlabel("explore probability (%0.2f - %0.2f)" % (min(explores), max(explores)), fontsize=20)
    pylab.ylabel("pheromone decay (%0.2f-%0.2f)" % (min(decays), max(decays)), fontsize=20)
    #pylab.title(title_str)
    #pylab.savefig(outname + '.png', format="png", transparent=True, bbox_inches='tight')
    pylab.savefig(outname + '.pdf', format='pdf', transparent=True, bbox_inches='tight')
    pylab.close()
    #print 'convert %s.png %s.pdf' % (outname, outname)
    #os.system('convert %s.png %s.pdf' % (outname, outname))

def ml_heat(label, sheets, strategies, decay_types, dmin=0.05, dmax=0.95, emin=0.05, \
            emax=0.95, dstep=0.05, estep=0.05, cumulative=False, out=False, ghost=False):
    decays = np.arange(dmin, dmax + dstep, dstep)
    explores = np.arange(emin, emax + estep, estep)
    hist_file = open('%s/repair_ml_hist.csv' % OUT_DIR, 'a')
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
                num_lines = sum(1 for line in open('%s/reformated_counts%s.csv' % (DATASETS_DIR, sheet)))
                total_lines += num_lines
                likelihoods = likelihood_matrix(sheet, explores, decays, likelihood_func,\
                                                decay_type, ghost)
                max_likelihood, max_values = max_likelihood_estimates(likelihoods, decays, explores)
                outname = '%s_%s' % (out_str, sheet)
                
                if cumulative:
                    if not isinstance(cumulative_likelihoods, np.ndarray):
                        cumulative_likelihoods = np.copy(likelihoods)
                    else:
                        cumulative_likelihoods += likelihoods
                        
                if IND_PLOT:
                    os.system('mkdir -p %s/individual/%s' % (OUT_DIR, stragegy))
                    plot_likelihood_heat(likelihoods, max_likelihood, max_values, explores, \
                                          decays, '%s/individual/%s/%s' % (OUT_DIR, strategy, outname))
                if out:
                    for explore, decay in max_values:
                        hist_file.write('%0.2f, %0.2f, %s, %s, %s, %d\n' % \
                                        (explore, decay, strategy, decay_type, label, num_lines))
            
            if cumulative:
                cumulative_likelihoods
                outname = 'cumulative_%s_%s' % (out_str, label)
                max_likelihood, max_values = max_likelihood_estimates(cumulative_likelihoods,\
                                                              decays, explores)
                os.system('mkdir -p %s/cumulative/%s' % (OUT_DIR, strategy))
                plot_likelihood_heat(cumulative_likelihoods, max_likelihood, max_values, \
                                explores, decays, '%s/cumulative/%s/%s' % (OUT_DIR, strategy, outname))
                print "plotted"
    hist_file.close()
                
    
if __name__ == '__main__':
    #strategy_choices = ['uniform', 'max', 'maxz', 'rank']
    #decay_choices = ['linear', 'const', 'exp']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', required=True)
    parser.add_argument('-sh', '--sheets', nargs='+', required=True)
    parser.add_argument('-s', '--strategies', action='store', nargs='+', \
                        choices=STRATEGY_CHOICES, required=True)
    parser.add_argument('-d', '--decay_types', nargs='+', choices=DECAY_CHOICES, required=True)
    parser.add_argument('-c', '--cumulative', action='store_true')
    parser.add_argument('-dmin', type=float, default=0.05)
    parser.add_argument('-dmax', type=float, default=0.95)
    parser.add_argument('-emin', type=float, default=0.05)
    parser.add_argument('-emax', type=float, default=0.95)
    parser.add_argument('-dstep', type=float, default=0.05)
    parser.add_argument('-estep', type=float, default=0.05)
    parser.add_argument('-o', '--out', action='store_true')
    parser.add_argument('-g', '--ghost', action='store_true')
    
    args = parser.parse_args()
    label = args.label
    sheets = args.sheets
    if sheets == ['nocut']:
        sheets = NOCUT
    elif sheets == ['nocut2']:
        sheets = NOCUT2
    elif sheets == ['cut']:
        sheets = CUT
    elif sheets == ['all']:
        sheets = ALL
    elif sheets == ['all2']:
        sheets = ALL2
    strategies = args.strategies
    decay_types = args.decay_types
    dmin, dmax, emin, emax = args.dmin, args.dmax, args.emin, args.emax
    dstep, estep = args.dstep, args.estep
    cumulative = args.cumulative
    out = args.out
    ghost = args.ghost
    if ghost:
        label += '_ghost'
    
    ml_heat(label, sheets, strategies, decay_types, dmin, dmax, emin, emax, dstep, estep, \
            cumulative, out, ghost)
