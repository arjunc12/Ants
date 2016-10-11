import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from sys import argv
import pandas as pd
import numpy as np
import pylab
import argparse
import numpy.ma as ma


MIN_PHEROMONE = 0
INIT_WEIGHT = 0
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

def check_graph(G):
    for u, v in G.edges_iter():
        weight = G[u][v]['weight']
        assert weight >= MIN_PHEROMONE
        wt = 0
        for unit in G[u][v]['units']:
            assert unit > MIN_PHEROMONE
            wt += unit
        assert wt == weight

def edge_weight(G, u, v):
    return sum(G[u][v]['units'])

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
    assert wt >= MIN_PHEROMONE
    G[u][v]['weight'] = x
    
def decay_graph_linear(G, decay, seconds=1):
    for u, v in G.edges_iter():
        decay_amount = decay * seconds
        decay_units(G, u, v, decay_amount)
        wt = edge_weight(G, u, v)
        assert wt >= MIN_PHEROMONE
        G[u][v]['weight'] = wt

def decay_graph_exp(G, decay, seconds=1):
    assert decay > 0
    assert decay < 1
    decay_prop = (1 - decay) ** seconds
    for u, v in G.edges_iter():
        G[u][v]['weight'] *= decay_prop
        assert G[u][v]['weight'] >= MIN_PHEROMONE

def get_decay_func(decay_type):
    if decay_type == 'const':
        return decay_graph_const
    elif decay_type == 'linear':
        return decay_graph_linear
    elif decay_type == 'exp':
        return decay_graph_exp
    else:
        raise ValueError("Invalid Decay Type")

def get_likelihood(choices, decay, explore, likelihood_func, decay_type, G=None):
    assert 0 < decay < 1
    df = pd.read_csv(choices, header=None, names=['source', 'dest', 'dt'])
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
    
    log_likelihood = 0
    G[sources[1]][dests[1]]['weight'] += 1
    if decay_type == 'linear':
        G[sources[1]][dests[1]]['units'].append(1)
    G2 = G.copy()
    for i in xrange(1, len(sources)):
        source = sources[i]
        dest = dests[i]
        
        log_likelihood += np.log(likelihood_func(G, source, dest, explore))
        if likelihood == 0:
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
               
    return np.log(likelihood), G

def get_max_likelihoods(likelihood_matrix, decays, explores):
    max_likelihood = float("-inf")
    max_values = []
    pos = 0
    for decay in decays:
        for explore in explores:
            i, j = pos / len(explores), pos % len(explores)
            likelihood = likelihoods[i, j]
            if likelihood > max_likelihood:
                max_values = [(explore, decay)]
                max_likelihood = likelihood
            elif max_likelihood == likelihood:
                max_values.append((explore, decay))
            pos += 1
    return max_likelihood, max_values
    
def get_likelihood_matrix(sheet, decay, explore, delta, likelihood_func, decay_type):
    delta = 0.05
    decays = np.arange(delta, 1, delta)
    explores = np.arange(delta, 1, delta)
    likelihoods = pylab.zeros((len(decays), len(explores)))
    G = None
    choices = 'reformated_counts%s.csv' % sheet
    pos = 0

    for decay in decays:
        for explore in explores:
            i, j = pos / len(explores), pos % len(explores)
            likelihood, G = decay_likelihood(choices, decay, explore, likelihood_func, G)
            #likelihood = pos
            likelihoods[i, j] = likelihood
            pos += 1
    
    max_likelihood, max_values = get_max_likelihoods(likelihoods, decays, explores)        
    likelihoods = ma.masked_invalid(likelihoods)
    return likelihoods, max_likelihood, max_values

def uniform_likelihood(G, source, dest, explore):
    w = G[source][dest]['weight']
    total = 0.0
    explored = 0
    unexplored = 0
    for n in G.neighbors(source):
        wt = G[source][n]['weight']
        if wt == MIN_PHEROMONE:
            unexplored += 1
        else:
            explored += 1
            total += wt
    if explored == 0:
        return 1.0 / unexplored
    elif w == MIN_PHEROMONE:
        return explore / unexplored
    else:
        prob = w / total
        return (1 - explore) * prob
    
def threshold_likelihood(G, source, dest, explore):
    above = []
    below = []
    wt = G[source][dest]['weight']
    above_total = 0.0
    for n in G.neighbors(source):
        w = G[source][n]['weight']
        if w >= THRESHOLD:
            above.append(n)
            above_total += w
        else:
            below.append(n)
    if dest in above:
        return (1 - explore) * (wt / above_total)
    else:
        return explore * (1.0 / len(below))
        
def max_edge_likelihood(G, source, dest, explore):
    max_wt = 0
    max_neighbors = []
    neighbors = G.neighbors(source)
    total = 0.0
    explored = 0
    unexplored = 0
    chosen_wt = G[source][dest]['weight']
    for n in neighbors:
        wt = G[source][n]['weight']
        if wt == MIN_PHEROMONE:
            unexplored += 1
        else:
            explored += 1
            total += wt
            if wt > max_wt:
                max_wt = wt
                max_neighbors = [n]
            elif wt == max_wt:
                max_neighbors.append(n)
    if explored == 0:
        return 1.0 / unexplored
    if dest in max_neighbors:
        return (1 - explore) / len(max_neighbors)
    else:
        return explore / (len(neighbors) - len(max_neighbors))
        
def maxz_edge_likelihood(G, source, dest, explore):
    max_wt = 0
    max_neighbors = []
    zero_neighbors = []
    neighbors = G.neighbors(source)
    for n in neighbors:
        wt = G[source][n]['weight']
        if wt == MIN_PHEROMONE:
            zero_neighbors.append(n)
        else:
            if wt > max_wt:
                max_wt = wt
                max_neighbors = [n]
            elif wt == max_wt:
                max_neighbors.append(n)
                
    if dest in max_neighbors:
        return (1 - explore) / len(max_neighbors)
    elif dest in zero_neighbors:
        return explore / (len(zero_neighbors))
    else:
        return 0      

def likelihood_heat(sheets, likelihood_func, strategy, outname):
    for sheet in sheets:
        print sheet
        pylab.figure()
        delta = 0.05
        decays = np.arange(0.05, 1, delta)
        explores = np.arange(0.05, 1, delta)
        likelihoods = pylab.zeros((len(decays), len(explores)))
        G = None
        choices = 'reformated_counts%s.csv' % sheet
        pos = 0

        for decay in decays:
            for explore in explores:
                i, j = pos / len(explores), pos % len(explores)
                likelihood, G = decay_likelihood(choices, decay, explore, likelihood_func, G)
                #likelihood = pos
                likelihoods[i, j] = likelihood
                pos += 1
                
        min_likelihood = float("inf")
        max_likelihood = float("-inf")
        bad_positions = []
        max_values = []
        pos = 0
        for decay in decays:
            for explore in explores:
                i, j = pos / len(explores), pos % len(explores)
                likelihood = likelihoods[i, j]
                if likelihood == float("-inf"):
                    bad_positions.append((i, j))
                else:
                    min_likelihood = min(likelihood, min_likelihood)
                    if likelihood > max_likelihood:
                        max_values = [(explore, decay)]
                        max_likelihood = likelihood
                    elif max_likelihood == likelihood:
                        max_values.append((explore, decay))
                pos += 1
        '''
        for i, j in bad_positions:
            likelihoods[i, j] = min_likelihood
        '''
        likelihoods = ma.masked_invalid(likelihoods)
            
        out_file = open('decay_ml.csv', 'a')
        title_str = ['max likelihood %f at:' % max_likelihood]
        for explore, decay in max_values:
            title_str.append('(e=%0.2f, d=%0.2f)' % (explore, decay))
            out_str = ', '.join([str(explore), str(decay), strategy, outname])
            out_file.write('%s\n' % out_str)
        out_file.close()
            
        title_str = '\n'.join(title_str)

        hm = pylab.pcolormesh(likelihoods, cmap='nipy_spectral')
        cb = pylab.colorbar(hm)
        cb.ax.set_ylabel('log-likelihood')
        pylab.tick_params(which='both', bottom='off', top='off', left='off', right='off', \
        labeltop='off', labelbottom='off', labelleft='off', labelright='off')
        
        pylab.xlabel("explore probability (%0.2f - %0.2f)" % (min(explores), max(explores)))
        pylab.ylabel("pheromone decay (%0.2f-%0.2f)" % (min(decays), max(decays)))
        pylab.title(title_str)
        pylab.savefig("decay_ml_%s_%s%s.png" % (strategy, outname, sheet), format="png", transparent=True,bbox_inches='tight')
        pylab.close()
        print "plotted"
        
def cumulative_likelihood_heat(sheets, likelihood_func, strategy, outname):
    delta = 0.05
    decays = np.arange(delta, 1, delta)
    explores = np.arange(delta, 1, delta)
    likelihoods = pylab.zeros((len(decays), len(explores)))
    denominator = 0
    pylab.figure()
    for sheet in sheets:
        print sheet
        G = None
        choices = 'reformated_counts%s.csv' % sheet
        f = open(choices)
        num_lines = sum(1 for line in f)
        denominator += num_lines
        f.close()
        pos = 0
        for decay in decays:
            for explore in explores:
                i, j = pos / len(explores), pos % len(explores)
                if likelihoods[i, j] == float("-inf"):
                    continue
                likelihood, G = decay_likelihood(choices, decay, explore, likelihood_func, G)
                #likelihood = pos
                likelihoods[i, j] += likelihood# * num_lines
                #print likelihoods
                pos += 1
    #likelihoods /= denominator
    min_likelihood = float("inf")
    max_likelihood = float("-inf")
    bad_positions = []
    max_values = []
    pos = 0
    for decay in decays:
        for explore in explores:
            i, j = pos / len(explores), pos % len(explores)
            likelihood = likelihoods[i, j]
            if likelihood == float("-inf"):
                bad_positions.append((i, j))
            else:
                min_likelihood = min(likelihood, min_likelihood)
                if likelihood > max_likelihood:
                    max_values = [(explore, decay)]
                    max_likelihood = likelihood
                elif max_likelihood == likelihood:
                    max_values.append((explore, decay))
            pos += 1
    
    '''        
    for i, j in bad_positions:
        likelihoods[i, j] = min_likelihood
    '''
    likelihoods = ma.masked_invalid(likelihoods)
    
    title_str = ['max likelihood %f at:' % max_likelihood]
    for explore, decay in max_values:
        title_str.append('(e=%0.2f, d=%0.2f)' % (explore, decay))
    title_str = '\n'.join(title_str)
            
    #print likelihoods
    hm = pylab.pcolormesh(likelihoods, cmap='nipy_spectral')
    cb = pylab.colorbar(hm)
    cb.ax.set_ylabel('log-likelihood')
    pylab.tick_params(which='both', bottom='off', top='off', left='off', right='off', \
    labeltop='off', labelbottom='off', labelleft='off', labelright='off')
    
    pylab.xlabel("explore probability (%0.2f-%0.2f)" % (min(explores), max(explores)))
    pylab.ylabel("pheromone decay (%0.2f-%0.2f)" % (min(decays), max(decays)))
    pylab.title(title_str)
    pylab.savefig("cumulative_decay_ml_%s_%s.png" % (strategy, outname), format="png", transparent=True,bbox_inches='tight')
    pylab.close()
    print "plotted"
    
def unif_likelihood_heat(sheets, outname):
    likelihood_heat(sheets, uniform_likelihood, 'uniform', outname)
    
def threshold_likelihood_heat(sheets, outname):
    likelihood_heat(sheets, threshold_likelihood, 'threshold', outname)
    
def max_edge_likelihood_heat(sheets, outname):
    likelihood_heat(sheets, max_edge_likelihood, 'max_edge', outname)
    
def maxz_edge_likelihood_heat(sheets, outname):
    likelihood_heat(sheets, maxz_edge_likelihood, 'maxz_edge', outname)
    
def likelihood_3dplot(sheets, likelihood_func, strategy):
    for sheet in sheets:
        sheet = int(sheet)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        delta = 0.02
        decays = np.arange(delta, 1, delta)
        explores = np.arange(delta, 1, delta)
        G = None
        choices = 'reformated_counts%d.csv' % sheet
        x = []
        y = []
        z = []
        max_likelihood = -float("inf")
        max_decay = None
        max_explore = None
        for decay in decays:
            for explore in explores:
                x.append(decay)
                y.append(explore)
                likelihood, G = decay_likelihood(choices, decay, explore, likelihood_func, G)
                if likelihood > max_likelihood:
                    max_likelihood = likelihood
                    max_decay = decay
                    max_explore = explore
                z.append(likelihood)
        ax.scatter(x, y, z)
        ax.set_xlabel('decay')
        ax.set_ylabel('explore')
        ax.set_zlabel('log-likelihood')
        pylab.title('maximum at decay = %f, explore = %f' % (max_decay, max_explore))
        pylab.savefig("decay_ml3d_%s%d.png" % (strategy, sheet), format="png")
        pylab.close()
        print "plotted"
        
def cumulative_unif_likelihood_heat(sheets, outname):
    cumulative_likelihood_heat(sheets, uniform_likelihood, 'uniform', outname)
    
def cumulative_max_edge_likelihood_heat(sheets, outname):
    cumulative_likelihood_heat(sheets, max_edge_likelihood, 'max_edge', outname)
    
def cumulative_maxz_edge_likelihood_heat(sheets, outname):
    cumulative_likelihood_heat(sheets, maxz_edge_likelihood, 'maxz_edge', outname)

def unif_likelihood_3dplot(sheets):
    likelihood_3dplot(sheets, uniform_likelihood, 'uniform')
    
def threshold_likelihood_3dplot(sheets):
    likelihood_3dplot(sheets, threshold_likelihood, 'threshold')
    
def max_edge_likelihood_3dplot(sheets):
    likelihood_3dplot(sheets, max_edge_likelihood, 'max_edge')
    
def maxz_edge_likelihood_3dplot(sheets):
    likelihood_3dplot(sheets, maxz_edge_likelihood, 'maxz_edge')
    
def cumulative_likelihood_3dplot(sheets, likelihood_func, strategy, outname):
    delta = 0.05
    decays = np.arange(delta, 1, delta)
    explores = np.arange(delta, 1, delta)
    x = []
    y = []
    for decay in decays:
        for explore in explores:
            x.append(decay)
            y.append(explore)
    z = np.zeros(len(x))
    denominator = 0
    for sheet in sheets:
        sheet = sheet
        print sheet
        choices = 'reformated_counts%s.csv' % sheet    
        f = open(choices)
        num_lines = sum(1 for line in f)
        denominator += num_lines
        f.close()
        G = None
        i = 0
        for decay in decays:
            for explore in explores:
                likelihood, G = decay_likelihood(choices, decay, explore, likelihood_func, G)
                likelihood *= num_lines
                z[i] += likelihood
                i += 1
    z /= denominator
    max_index = np.argmax(z)
    max_decay = x[max_index]
    max_explore = y[max_index]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('decay')
    ax.set_ylabel('explore')
    ax.set_zlabel('log-likelihood')
    pylab.title('maximum at decay = %f, explore = %f' % (max_decay, max_explore))
    pylab.savefig("cumulative_decay_ml3d_%s_%s.png" % (outname, strategy), format="png")
    pylab.close()
    print "plotted"
    
def cumulative_unif_likelihood_3dplot(sheets, outname):
    cumulative_likelihood_3dplot(sheets, uniform_likelihood, 'uniform', outname)
    
def cumulative_max_edge_likelihood_3dplot(sheets, outname):
    cumulative_likelihood_3dplot(sheets, max_edge_likelihood, 'max_edge', outname)
    
def cumulative_maxz_edge_likelihood_3dplot(sheets, outname):
    cumulative_likelihood_3dplot(sheets, maxz_edge_likelihood, 'maxz_edge', outname)
        
def likelihood_2dplot(sheets, likelihood_func, strategy, explore=0.1):
    for sheet in sheets:
        sheet = int(sheet)
        fig = plt.figure()
        delta = 0.01
        decays = np.arange(delta, 1, delta)
        G = None
        choices = 'reformated_counts%s.csv' % sheet
        likelihoods = []
        max_likelihood = -float("inf")
        max_decay = None
        for decay in decays:
            likelihood, G = decay_likelihood(choices, decay, explore, likelihood_func, G)
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                max_decay = decay
            likelihoods.append(likelihood)
        
        pylab.plot(decays, likelihoods)
        pylab.title('maximum at decay = %f' % max_decay)
        pylab.savefig("decay_ml2d_%s%s.png" % (strategy, sheet), format="png")
        pylab.close()
        print "plotted"
        
def unif_likelihood_2dplot(sheets):
    likelihood_2dplot(sheets, uniform_likelihood, 'uniform')
    
def threshold_likelihood_2dplot(sheets):
    likelihood_2dplot(sheets, threshold_likelihood, 'threshold')
    
def max_edge_likelihood_2dplot(sheets):
    likelihood_2dplot(sheets, max_edge_likelihood, 'max_edge')
    
def maxz_edge_likelihood_2dplot(sheets):
    likelihood_2dplot(sheets, maxz_edge_likelihood, 'maxz_edge')

def get_likelihood_func(strategy):
    if strategy == 'uniform':
        return uniform_likelihood
    elif strategy == 'max':
        return max_edge_likelihood
    elif strategy == 'maxz':
        return maxz_edge_likelihood
    else:
        raise ValueError('invalid strategy')

def make_title_str(max_likelihood, max_values):
    title_str = ['max likelihood %f at:' % max_likelihood]
    for explore, decay in max_values:
        title_str.append('(e=%0.2f, d=%0.2f)' % (explore, decay))
    title_str = '\n'.join(title_str)
    return title_str

def plot_likelihood_heat(likelihoods, max_likelihood, max_values, explores, decays, outname):
    title_Str = make_title_str(max_likelihood, max_values)
    pylab.figure()
    hm = pylab.pcolormesh(likelihoods, cmap='nipy_spectral')
    cb = pylab.colorbar(hm)
    cb.ax.set_ylabel('log-likelihood')
    pylab.tick_params(which='both', bottom='off', top='off', left='off', right='off', \
    labeltop='off', labelbottom='off', labelleft='off', labelright='off')
    
    pylab.xlabel("explore probability (%0.2f - %0.2f)" % (min(explores), max(explores)))
    pylab.ylabel("pheromone decay (%0.2f-%0.2f)" % (min(decays), max(decays)))
    pylab.title(title_str)
    pylab.savefig(outname, format="png", transparent=True,bbox_inches='tight')
    pylab.close()
    print "plotted"

def ml_heat(label, sheets, strategies, decay_types, delta=0.05, cumulative=False):
    for strategy in strategies:
        likelihood_func = get_likelihood_func(strategy)
        for decay_type in decay_types:
            out_str = 'repair_ml_%s_%s_%s' % (strategy, decay_type, label)
            cumulative_likelihoods = None
            for sheet in sheets:
                likelihoods, max_likelihood, max_values = get_likelihood_matrix(sheet, \
                                       decay_type, explore, delta, likelihood_func, decay_type)
                if cumulative:
                    if cumulative_likelihoods == None:
                        cumulative_likelihoods = np.copy(likelihoods)
                    else:
                        cumulative_likelihoods += likelihoods
                outname = '%s_%s.png' % (out_str, sheet)   
                plot_likelihood(likelihoods, max_likelihood, max_values, explores, \
                                decays, outname)
            if cumulative:
                outname = 'cumulative_%s.png' % out_str
                max_likelihoods, max_values = \
                             get_max_likelihoods(cumulative_likelihoods, decays, explores)
                plot_likelihood(cumulative_likelihoods, max_likelihood, max_values, \
                                explores, decays, outname)
                
    
if __name__ == '__main__':
    #outname = argv[1]
    #sheets = argv[2:]
    
    strategy_choices = ['uniform', 'max', 'maxz']
    decay_choices = ['linear', 'const', 'exp']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('sheets', nargs='+')
    parser.add_argument('-s', '--strategies', action='store', nargs='+', \
                        choices=strategy_choices, required=True)
    parser.add_argument('-d', '--decay_types', nargs='+', choices=decay_choices, required=True)
    parser.add_argument('-dt', '--delta', type=float, default=0.05)
    parser.add_argument('-c', '--cumulative', action='store_true')
    
    args = parser.parse_args()
    label = args.label
    sheets = args.sheets
    strategies = args.strategies
    decay_types = args.decay_types
    delta = args.delta
    cumulative = args.cumulative
    
    print label, sheets, strategies, decays, delta, cumulative
        
    #unif_likelihood_heat(sheets, outname)
    #max_edge_likelihood_heat(sheets, outname)
    #maxz_edge_likelihood_heat(sheets, outname)
    
    #unif_likelihood_3dplot(sheets, outname)
    #max_edge_likelihood_3dplot(sheets, outname)
    
    #unif_likelihood_2dplot(sheets, outname)
    #max_edge_likelihood_2dplot(sheets, outname)
    
    #cumulative_unif_likelihood_3dplot(sheets, outname)
    #cumulative_max_edge_likelihood_3dplot(sheets, outname)
    
    #cumulative_unif_likelihood_heat(sheets, outname)
    #cumulative_max_edge_likelihood_heat(sheets, outname)
    #cumulative_maxz_edge_likelihood_heat(sheets, outname)