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
from choice_functions import dberg_likelihood
from decay_functions import *
import os
from collections import defaultdict

MIN_PHEROMONE = 0
MIN_DETECTABLE_PHEROMONE = 0
INIT_WEIGHT = 0

IND_PLOT = False

NOCUT = ['6', '7a', '7c', '8a', '8b', '9b', '14','15', '16', '17', '21a', '21b', '23a', '23d']
CUT = ['7b', '9a', '9c', '9d', '10', '11', '12', '13', '18', '19', '20', '21c', '22', '23b', '23c', '23e', '23f']
ALL = NOCUT + CUT

NOCUT2 = ['6', '7a', '7c', '9b', '14','15', '16', '21a', '21b', '23a', '23d']
ALL2 = NOCUT2 + CUT

STRATEGY = 'dberg'
STRATEGIES = [STRATEGY]

datasets_dict = {'nocut' : NOCUT, 'nocut2' : NOCUT2, 'cut' : CUT, 'all' : ALL, 'all2' : ALL2}

DATASETS_DIR = 'datasets/reformated_csv'
OUT_DIR = 'ml_plots'
ML_OUTFILE = '%s/dberg_ml.csv' % OUT_DIR

DECAY_RATE = 0.01
DECAY_TYPE = 'exp'

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
        max_degree = 0
        for u in G.nodes_iter():
            max_degree = max(max_degree, G.degree(u))

        for u in G.nodes():
            if G.degree(u) == 1:
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

def param_likelihood(choices, exponent, offset, decay=DECAY_RATE,\
                     decay_type=DECAY_TYPE, G=None, ghost=False):
    assert 0 <= decay <= 1
    explore = 1.0 / exponent
    #assert 0 <= explore <= 1
    df = pd.read_csv(choices, header=None, names=['source', 'dest', 'dt'], skipinitialspace=True)
    df['dt'] = pd.to_datetime(df['dt'])
    df.sort_values(by='dt', inplace=True)
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
    
    '''
    G[sources[0]][dests[0]]['weight'] += 1
    if decay_type == 'linear':
        G[sources[0]][dests[0]]['units'].append(1)
    '''

    G2 = G.copy()
    
    max_degree = 0
    for u in G.nodes():
        max_degree = max(max_degree, len(G.neighbors(u)))
    
    for i in xrange(0, len(sources)):
        #check_graph(G, check_units)
        #check_graph(G2, check_units)
        
        source = sources[i]
        dest = dests[i]
        
        log_likelihood += np.log(dberg_likelihood(G, source, dest, explore=explore, offset=offset))
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
        add_amount = 1
        
        if G[source][dest]['weight'] <= MIN_DETECTABLE_PHEROMONE:
            pass #add_amount *= 2
        G2[source][dest]['weight'] += add_amount
        if decay_type == 'linear':
            G2[source][dest]['units'].append(add_amount)

    return log_likelihood, G

def max_likelihood_estimates(likelihoods, exponents, offsets):
    max_likelihood = float("-inf")
    max_values = []
    pos = 0
    for offset in offsets:
        for exponent in exponents:
            i, j = pos / len(exponents), pos % len(exponents)
            pos += 1
            likelihood = likelihoods[i, j]
            if likelihood == float("-inf"):
                continue
            if likelihood > max_likelihood:
                max_values = [(exponent, offset)]
                max_likelihood = likelihood
            elif max_likelihood == likelihood:
                max_values.append((exponent, offset))
    return max_likelihood, max_values
    
def likelihood_matrix(sheet, exponents, offsets, decay_rate=DECAY_RATE, decay_type=DECAY_TYPE,\
                      ghost=False):
    likelihoods = pylab.zeros((len(offsets), len(exponents)))
    G = None
    choices = '%s/reformated_counts%s.csv' % (DATASETS_DIR, sheet)
    pos = 0

    for offset in offsets:
        for exponent in exponents:
            i, j = pos / len(exponents), pos % len(exponents)
            likelihood, G = param_likelihood(choices, exponent, offset,\
                                             decay=decay_rate,\
                                             decay_type=decay_type, G=G,\
                                             ghost=ghost)
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

def likelihood_plot(likelihoods, exponents, offsets, outname):
    pos = 0
    x = []
    y = []
    for offset in offsetss:
        best_exponents = []
        best_likelihood = float("-inf")
        for exponent in exponents:
            i, j = pos / len(exponents), pos % len(exponents)
            likelihood = likelihoods[i, j]
            pos += 1
            if np.isnan(likelihood):
                continue
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_exponents = [exponent]
            elif likelihood == best_likelihood:
                best_exponents.append(exponents)
        for best_exponent in best_exponents:
            x.append(offset)
            y.append(best_exponent)
                
    pylab.figure()
    pylab.scatter(x, y)
    pylab.xlabel('offset')
    pylab.ylabel('best exponent')
    pylab.savefig(outname + '_plot.pdf', format='pdf', transparent=True, bbox_inches='tight')
    pylab.close()
        
def likelihood_heat(likelihoods, max_likelihood, max_values, exponents,\
                         offsets, outname):
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
    
    pylab.xlabel("explore probability (%0.2f - %0.2f)" % (min(exponents), max(exponents)), fontsize=20)
    pylab.ylabel("pheromone decay (%0.2f-%0.2f)" % (min(offsets), max(offsets)), fontsize=20)
    #pylab.title(title_str)
    #pylab.savefig(outname + '.png', format="png", transparent=True, bbox_inches='tight')
    pylab.savefig(outname + '_heat.pdf', format='pdf', transparent=True, bbox_inches='tight')
    pylab.close()
    #print 'convert %s.png %s.pdf' % (outname, outname)
    #os.system('convert %s.png %s.pdf' % (outname, outname))

def write_likelihoods(likelihoods, sheet, num_lines, decay_type,\
                      decay_rate, ghost, exponents, offsets,\
                      outfile=ML_OUTFILE):
    f = open(outfile, 'a')
    pos = 0
    for offset in offsets:
        for exponent in exponents:
            i, j = pos / len(exponents), pos % len(exponents)
            likelihood = likelihoods[i, j]
            pos += 1
            f.write('%s, %d, %s, %s, %d, %f, %f, %f\n' % (sheet, num_lines,\
                                                          STRATEGY, decay_type,\
                                                          int(ghost), exponent,\
                                                          offset, likelihood))
    f.close()

def ml_analysis(label, sheets, decay_types, decay_rate=DECAY_RATE, omin=0.01, omax=2, emin=0.01, \
            emax=2, ostep=0.01, estep=0.01, cumulative=False, out=False, ghost=False,\
            heat=True, plot=False, write_file=False):
    offsets = np.arange(omin, omax + ostep, ostep)
    exponents = np.arange(emin, emax + estep, estep)
    hist_file = open('%s/dberg_ml_hist.csv' % OUT_DIR, 'a')
    for decay_type in decay_types:
        print decay_type
        out_str = 'dberg_ml_%s_%s' % (STRATEGY, decay_type)
        cumulative_likelihoods = None
        total_lines = 0
        for sheet in sheets:
            print sheet
            num_lines = sum(1 for line in open('%s/reformated_counts%s.csv' % (DATASETS_DIR, sheet)))
            total_lines += num_lines
            likelihoods = likelihood_matrix(sheet, exponents, offsets,\
                                            decay_rate=decay_rate,\
                                            decay_type=decay_type, ghost=ghost)
            if write_file:
                write_likelihoods(likelihoods, sheet, num_lines, decay_type,\
                                  decay_rate, ghost, exponents, offsets)

            max_likelihood, max_values = max_likelihood_estimates(likelihoods, exponents, offsets)
            outname = '%s_%s' % (out_str, sheet)
            
            if cumulative:
                if not isinstance(cumulative_likelihoods, np.ndarray):
                    cumulative_likelihoods = np.copy(likelihoods)
                else:
                    cumulative_likelihoods += likelihoods
                    
            if IND_PLOT:
                os.system('mkdir -p %s/individual/%s' % (OUT_DIR, stragegy))
                if heat:
                    likelihood_heat(likelihoods, max_likelihood, max_values, exponents, \
                                      offsets, '%s/individual/%s/%s' % (OUT_DIR, strategy, outname))
                if plot:
                    likelihood_plot(likelihoods, exponents, offsets, \
                                    '%s/individual/%s/%s' % (OUT_DIR, strategy, outname))
            if out:
                for exponent, offset in max_values:
                    hist_file.write('%0.2f, %0.2f, %s, %s, %s, %d\n' % \
                                    (exponent, offset, strategy, decay_type, label, num_lines))
        
        if cumulative:
            cumulative_likelihoods
            outname = 'cumulative_%s_%s' % (out_str, label)
            max_likelihood, max_values = max_likelihood_estimates(cumulative_likelihoods,\
                                                          decays, explores)
            os.system('mkdir -p %s/cumulative/%s' % (OUT_DIR, strategy))
            if heat:
                likelihood_heat(cumulative_likelihoods, max_likelihood, max_values, \
                            exponents, offsets, '%s/cumulative/%s/%s' % (OUT_DIR, strategy, outname))
            if plot:
                likelihood_plot(cumulative_likelihoods, exponents, offsets, \
                                '%s/cumulative/%s/%s' % (OUT_DIR, strategy, outname))
            print "plotted"
    hist_file.close()
                
   
def main():
    #strategy_choices = ['uniform', 'max', 'maxz', 'rank']
    #decay_choices = ['linear', 'const', 'exp']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', default=None)
    parser.add_argument('-sh', '--sheets', nargs='+', required=True)
    parser.add_argument('-d', '--decay', type=float, default=DECAY_RATE)
    parser.add_argument('-dt', '--decay_types', nargs='+', choices=DECAY_CHOICES, required=True)
    parser.add_argument('-c', '--cumulative', action='store_true')
   
    parser.add_argument('-omin', type=float, default=0.01)
    parser.add_argument('-omax', type=float, default=2)
    parser.add_argument('-emin', type=float, default=0.01)
    parser.add_argument('-emax', type=float, default=2)
    parser.add_argument('-ostep', type=float, default=0.01)
    parser.add_argument('-estep', type=float, default=0.01)
    
    parser.add_argument('-o', '--out', action='store_true')
    
    parser.add_argument('-g', '--ghost', action='store_true')
    
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--heat', action='store_true')
    parser.add_argument('--write_file', action='store_true')
    
    args = parser.parse_args()
    label = args.label
    sheets = args.sheets
    
    if label == None:
        assert len(sheets) == 1
        label = sheets[0]
    
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

    decay_types = args.decay_types
    decay_rate = args.decay
    omin, omax, emin, emax = args.omin, args.omax, args.emin, args.emax
    ostep, estep = args.ostep, args.estep
    cumulative = args.cumulative
    out = args.out
    ghost = args.ghost
    if ghost:
        label += '_ghost'
    
    heat = args.heat
    plot = args.plot
    write_file = args.write_file

    if not (heat or plot or write_file):
        print "error: must select an action"
        return None

    ml_analysis(label, sheets, decay_types, decay_rate, omin, omax, emin, emax, ostep, estep, \
                cumulative, out, ghost, heat, plot, write_file)

if __name__ == '__main__':
    main()
