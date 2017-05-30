import pandas as pd
import numpy as np
import argparse
from scipy.stats import gmean, hmean
import numpy.ma as ma
import matplotlib as mpl
mpl.use('agg')
import pylab
import os
from graphs import GRAPH_CHOICES
from choice_functions import STRATEGY_CHOICES
from decay_functions import DECAY_CHOICES
from plot_ant_repair import COLUMNS

EXPLORE_MLE = 0.2
DECAY_MLE = 0.02

def plot_aggregate_dfs(df, metrics, descriptor):
    '''
    Plots an aggregate heatmap. Given a heatmap with the performance of several metrics across
    several networks, this function aggregates the performance of each metric across the
    networks. Uses geometric mean to aggregate results for two reasons. First, for certain metrics
    the range of values for that metric can vary wildly across network topologies. The geometric mean
    (as opposed to arithmetic mean) corrects for this and prevents networks where the
    metric tends to take on higher values to have a disproportionate effect. Second, for
    success rate metrics that are [0, 1]-valued, the geometric mean strongly penalizes an algorithm
    if it performs extremely poorly on any one network
    
    '''
    x = df['explore'].unique()
    y = df['decay'].unique()
    matrices = []
    for i in xrange(len(metrics)):
        matrices.append(np.zeros((len(y), len(x))))
    pos = 0
    best_aggs = [float("-inf")] * len(metrics)
    best_explores = [None] * len(metrics)
    best_decays = [None] * len(metrics)
    for name, group in df.groupby(['explore', 'decay']):
        explore, decay = name
        i, j = pos % len(y), pos / len(y)
        for k, metric in enumerate(metrics):
            val = gmean(group[metric])
            if val > best_aggs[k]:
                best_aggs[k] = val
                best_explores[k] = explore
                best_decays[k] = decay
            if explore == EXPLORE_MLE and decay == DECAY_MLE:
                pass #print metric, val
            z = matrices[k]
            z[i, j] = val
        pos += 1
        
    if pos != len(x) * len(y):
        print pos, len(x), len(y), len(x) * len(y)
    assert pos == len(x) * len(y)
        
    for k, metric in enumerate(metrics):
        title = 'aggregate_%s' % metric
        z = matrices[k]
        z = ma.masked_invalid(z)
        pylab.figure()
        map = 'Reds'
        vmin = 0
        vmax = pylab.nanmax(pylab.absolute(z))
        cb_label = '%s mean' % metric

        hm = pylab.pcolormesh(z, cmap=map, vmin=vmin, vmax=vmax, clim=(vmin, vmax))
        curr_ax = pylab.gca()
        curr_ax.axis('tight')
        cb = pylab.colorbar(hm)
        cb.set_clim(vmin, vmax)
        cb.ax.set_ylabel('geometric mean', fontsize=20)
        cb.ax.set_ylim(bottom=0)
        pylab.tick_params(which='both', bottom='off', top='off', left='off', right='off', \
        labeltop='off', labelbottom='off', labelleft='off', labelright='off')
        pylab.xlabel("explore probability (%0.2f - %0.2f)" % (min(x), max(x)), fontsize=20)
        pylab.ylabel("pheromone decay (%0.2f-%0.2f)" % (min(y), max(y)), fontsize=20)
        pylab.savefig("%s_%s.png" % (title, descriptor), format="png", transparent=True, bbox_inches='tight')
        os.system('convert %s_%s.png %s_%s.pdf' % (title, descriptor, title, descriptor))
        pylab.close()

        print metric, best_aggs[k], best_explores[k], best_decays[k]

def main():
    columns = ['ants', 'explore', 'decay', 'has_path', 'cost', 'path_entropy', 'walk_entropy', \
               'mean_journey_time', 'median_journey_time', 'walk_success_rate', 'pruning',\
               'connect_time', 'path_pruning', 'chosen_path_entropy', 'walk_pruning', \
               'chosen_walk_entropy', 'wasted_edge_count', 'wasted_edge_weight', 'mean_path_len']
    
    '''
    graph_choices = ['fig1', 'full', 'simple', 'simple_weighted', 'simple_multi', \
                     'full_nocut', 'simple_nocut', 'small', 'tiny', 'medium', \
                     'medium_nocut', 'grid_span', 'grid_span2', 'grid_span3', 'er', \
                     'mod_grid', 'half_grid', 'mod_grid_nocut', 'half_grid_nocut',\
                     'mod_grid1', 'mod_grid2', 'mod_grid3', 'vert_grid', 'barabasi', \
                     'vert_grid1', 'vert_grid2', 'vert_grid3', 'caroad', 'paroad', \
                     'txroad', 'subelji']
    '''
    
    #strategy_choices = ['uniform', 'max', 'hybrid', 'maxz', 'hybridz', 'rank']
    #decay_choices = ['linear', 'const', 'exp']
    

    usage="usage: %prog [options]"
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graphs", dest='graphs', choices=GRAPH_CHOICES,\
                        nargs='+', required=True)
    parser.add_argument('-s', '--strategies', dest='strategies', choices=STRATEGY_CHOICES,\
                        required=True, nargs='+')
    parser.add_argument("-dt", "--decay_types", dest="decay_types", choices=DECAY_CHOICES,\
                        required=True, nargs='+')
    parser.add_argument('-m', '--max_steps', dest='max_steps', type=int, required=True)
    parser.add_argument('-l', '--label', dest='label', required=True)
    parser.add_argument('-me', '--metrics', dest='metrics', choices=COLUMNS, nargs='+', \
                        required=True)
    
    args = parser.parse_args()
    # ===============================================================

    # ===============================================================
    graphs = args.graphs
    strategies = args.strategies
    decay_types = args.decay_types
    metrics = args.metrics
    max_steps = args.max_steps
    label = args.label
    
    frames = []
    
    for strategy in strategies:
        print strategy
        for decay_type in decay_types:
            for graph in graphs:
                descriptor = '%s_%s_%s' % (strategy, graph, decay_type)
                filename = 'ant_repair_%s%d.csv' % (descriptor, max_steps)
                df = pd.read_csv(filename, header=None, names = COLUMNS,\
                                 na_values='nan', skipinitialspace=True)
                df = df[['explore', 'decay'] + metrics]
                df = df.groupby(['explore', 'decay'], as_index=False).agg(np.mean)
                frames.append(df)
        descriptor = '%s_%s%s' % (strategy, decay_type, label)
        df = pd.concat(frames)    
        plot_aggregate_dfs(df, metrics, descriptor)
                    

if __name__ == '__main__':
    main()
