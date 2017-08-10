import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import pylab
from sys import argv
import argparse
DEBUG = False
NARROW = False
COMPLETE = True
import numpy.ma as ma
import os

MLE_EXPLORE = 0.2
MLE_DECAY = 0.02

TRANSPARENT = False

COLUMNS = ['ants', 'explore', 'decay', 'has_path', 'cost', 'path_entropy', 'walk_entropy', \
           'mean_journey_time', 'median_journey_time', 'walk_success_rate', 'cost_pruning',\
           'connect_time', 'path_pruning', 'chosen_path_entropy', 'walk_pruning', \
           'chosen_walk_entropy', 'wasted_edge_count', 'wasted_edge_weight', 'mean_path_len']


def ant_repair_plot(df, strategy, graph, decay_type, group_func=pylab.mean):
    pass

def heat(df, group_func, title, strategy, cb_label, sequential=True, vmax=None):
    x = df['explore'].unique()
    y = df['decay'].unique()
    z = pylab.zeros((len(y), len(x)))
    grouped = df.groupby(['explore', 'decay'])
    pos = 0
    for name, group in grouped:
        val = group_func(group)
        if name == (MLE_EXPLORE, MLE_DECAY):
            pass #print title, val
        i, j = pos % len(y), pos / len(y)
        z[i, j] = val
        pos += 1
    if pos != len(x) * len(y):
        print pos, len(x), len(y), len(x) * len(y)
    assert pos == len(x) * len(y)
    z = ma.masked_invalid(z)
    pylab.figure()
    map = 'Reds'
    vmin = 0
    if vmax == None:
        vmax = pylab.nanmax(pylab.absolute(z))
    if vmax == vmin:
        vmax = vmin + 0.01
    if not sequential:
        map = 'coolwarm'
        vmin = -max_abs
    #print title, vmin, vmax, map
    hm = pylab.pcolormesh(z, cmap=map, vmin=vmin, vmax=vmax, clim=(vmin, vmax))
    curr_ax = pylab.gca()
    curr_ax.axis('tight')
    cb = pylab.colorbar(hm)
    cb.set_clim(vmin, vmax)
    cb.ax.set_ylabel(cb_label, fontsize=20)
    cb.ax.set_ylim(bottom=0)
    pylab.tick_params(which='both', bottom='off', top='off', left='off', right='off', \
    labeltop='off', labelbottom='off', labelleft='off', labelright='off')
    pylab.xlabel("explore probability (%0.2f - %0.2f)" % (min(x), max(x)), fontsize=20)
    pylab.ylabel("pheromone decay (%0.2f-%0.2f)" % (min(y), max(y)), fontsize=20)
    pylab.savefig("%s_%s.png" % (title, strategy), format="png", transparent=TRANSPARENT, bbox_inches='tight')
    os.system('convert %s_%s.png %s_%s.pdf' % (title, strategy, title, strategy))

def describe_group(group):
    print pylab.mean(group['explore']), pylab.mean(group['decay'])
    
def connect_time_heat(df, strategy):
    def connect_time(group):
        return pylab.nanmean(group['connect_time'])
    
    heat(df, connect_time, "connect_time", strategy, "average time taken to form a path between nests")
    
def connectivity_heat(df, strategy):
    def connectivity(group):
        return pylab.log10(pylab.nanmean(group['connectivity']))
    
    heat(df, connectivity, "connectivity", strategy, "average number of simple paths between nest and target")

def distance_heat(df, strategy):
    def distance(group):
        return pylab.nanmean(group['dist'])
        
    heat(df, distance, 'distance', strategy, 'average shortest path length from nest to target at the end')
    
def mean_dist_heat(df, strategy):
    def mean_dist(group):
        return pylab.nanmean(group['mean_dist'])
        
    heat(df, mean_dist, 'mean_distance', strategy, 'average distance from nest to target at the end')
    
def score_heat(df, strategy):
    def score(group):
        return pylab.nanmean(group['score'])
        
    heat(df, score, 'paths_score', strategy, 'average ratio of path weight to path length')

def correlation_heat(df, strategy):
    def correlation(group):
        return pylab.nanmean(group['correlation'])
        
    heat(df, correlation, 'path_correlation', strategy, 'negative correlation of path length and path weight/edge')

def cost_heat(df, strategy):
    def cost(group):
        return pylab.nanmean(group['cost'])
        
    heat(df, cost, 'graph_cost', strategy, 'cost of pheromone subgraph')
    
def node_entropy_heat(df, strategy):
    def node_entropy(group):
        return pylab.nanmean(group['node_etr'])
        
    heat(df, node_entropy, 'node_entropy', strategy, 'average vertex entropy in pheromone subgraph')
    
def min_entropy_heat(df, strategy):
    def min_entropy(group):
        return pylab.nanmean(group['min_etr'])
        
    heat(df, min_entropy, 'min_entropy', strategy, 'entropy of lowest entropy path')

def mean_entropy_heat(df, strategy):
    def mean_entropy(group):
        return pylab.nanmean(group['mean_etr'])
        
    heat(df, mean_entropy, 'mean_entropy', strategy, 'average path entropy')
    
def path_entropy_heat(df, strategy):
    def path_entropy(group):
        if all(pylab.isnan(group['path_entropy'])):
            pass
            #print "path entropy"
            #describe_group(group)
        return pylab.nanmean(group['path_entropy'])
        
    heat(df, path_entropy, 'path_entropy', strategy, 'path entropy')
    
def min_entropy_dist_heat(df, strategy):
    def min_entropy_dist(group):
        return pylab.nanmean(group['min_etr_dist'])
        
    heat(df, min_entropy_dist, 'min_entropy_dist', strategy, 'length of minimum entropy path')
    
def mean_journey_heat(df, strategy):
    def mean_journey_time(group):
        return pylab.nanmean(group['mean_journey_time'])
        
    heat(df, mean_journey_time, 'mean_journey_time', strategy, 'average time for new ants to find nest', sequential=True)
    
def med_journey_heat(df, strategy):
    def med_journey_time(group):
        return pylab.nanmedian(group['mean_journey_time'])
        
    heat(df, med_journey_time, 'median_journey_time', strategy, 'median time for new ants to find nest', sequential=True)
    
def popular_len_heat(df, strategy):
    def mean_popular_len(group):
        return pylab.nanmean(group['popular_len'])
        
    heat(df, mean_popular_len, 'mean_popular_len', strategy, 'average length of most popular path')
    
def walk_entropy_heat(df, strategy):
    def mean_walk_entropy(group):
        if all(pylab.isnan(group['walk_entropy'])):
            print "walk entropy"
            describe_group(group)
        return pylab.nanmean(group['walk_entropy'])
        
    heat(df, mean_walk_entropy, 'walk_entropy', strategy, 'entropy over all chosen walks', sequential=True)
    
def path_success_rate_heat(df, strategy):
    def path_success_rate(group):
        success_rate = pylab.nanmean(group['has_path'])
        if success_rate == 0:
            pass
            #print "path success rate"
            #describe_group(group)
        return success_rate
    heat(df, path_success_rate, 'path_success_rate', strategy, \
         'success rate', sequential=True, vmax=1)
    
def walk_success_rate_heat(df, strategy):
    def walk_success_rate(group):
        success_rate = pylab.nanmean(group['walk_success_rate'])
        if success_rate == 0:
            print "walk success rate"
            describe_group(group)
        return success_rate
    heat(df, walk_success_rate, 'walk_success_rate', strategy, 'proportion successful walks for new ants', sequential=True)

def pruning_heat(df, strategy):
    def pruning(group):
        return pylab.nanmean(group['pruning'])
        
    heat(df, pruning, 'pruning', strategy, 'average edge pruning done between start and end')

def cost_pruning_heat(df, strategy):
    def pruning(group):
        return pylab.nanmean(group['cost_pruning'])
        
    heat(df, pruning, 'cost_pruning', strategy, 'average edge pruning done between start and end')

def path_pruning_heat(df, strategy):
    def path_pruning(group):
        if all(pylab.isnan(group['path_pruning'])):
            print "path pruning"
            describe_group(group)
            
        return pylab.nanmean(group['path_pruning'])
            
    heat(df, path_pruning, 'path_pruning', strategy, 'average path pruning done between start and end')

def chosen_path_entropy_heat(df, strategy):
    def chosen_path_entropy(group):
        if all(pylab.isnan(group['chosen_path_entropy'])):
            print "chosen_path_entropy"
            describe_group(group)
            
        return pylab.nanmean(group['chosen_path_entropy'])
            
    heat(df, chosen_path_entropy, 'chosen_path_entropy', strategy, 'average entropy over chosen paths')

def walk_pruning_heat(df, strategy):
    def walk_pruning(group):
        if all(pylab.isnan(group['walk_pruning'])):
            print "walk pruning"
            describe_group(group)
            
        return pylab.nanmean(group['walk_pruning'])
            
    heat(df, walk_pruning, 'walk_pruning', strategy, 'average walk pruning done between start and end')

def chosen_walk_entropy_heat(df, strategy):
    def chosen_walk_entropy(group):
        if all(pylab.isnan(group['chosen_walk_entropy'])):
            print "chosen_walk_entropy"
            describe_group(group)
            
        return pylab.nanmean(group['chosen_walk_entropy'])
            
    heat(df, chosen_walk_entropy, 'chosen_walk_entropy', strategy, 'average entropy over chosen walks')

def wasted_edge_count_heat(df, strategy):
    make_heat(df, strategy, 'wasted_edge_count', 'number of edges not contributing to any path')
    
def wasted_edge_weight_heat(df, strategy):
    make_heat(df, strategy, 'wasted_edge_weight', 'weight of edges not contributing to any path')

def mean_path_len_heat(df, strategy):
    make_heat(df, strategy, 'mean_path_len', 'average path length')
    
def var_path_len_heat(df, strategy):
    make_heat(df, strategy, 'mean_path_len', 'path length variance',\
             'var_path_len', lambda x : pylab.nanvar(x, ddof=1))
             
def std_path_len_heat(df, strategy):
    make_heat(df, strategy, 'mean_path_len', 'path length standard deviation',\
             'std_path_len', lambda x : pylab.nanstd(x, ddof=1))
    
def make_heat(df, strategy, metric, description, metric_name=None, groupfunc=pylab.nanmean):
    if metric_name == None:
        metric_name = metric 
    def metric_heat(group):
        if all(pylab.isnan(group[metric])):
            #print metric
            #describe_group(group)
            pass
           
        return groupfunc(group[metric])
        
    heat(df, metric_heat, metric_name, strategy, description)
    
def main():
    filename = argv[1]
    strategy = argv[2]
    
    
    df = pd.read_csv(filename, header=None, names = COLUMNS, na_values='nan', skipinitialspace=True)
    
    if DEBUG:
        pos = 0
        for name, group in df.groupby(['explore', 'decay']):
            print name
            explore, decay = name
            if not (0.01 <= explore <= 0.2):
                print "----------BAD EXPLORE-----------"
            if not (0.01 <= decay <= 0.2):
                print "----------BAD DECAY-------------"
            pos += 1
        l1 = len(df['explore'].unique())
        l2 = len(df['decay'].unique())
        print l1, l2, l1 * l2, pos
        return None
    
    
    path_success_rate_heat(df, strategy)
    path_entropy_heat(df, strategy)
    
    #wasted_edge_count_heat(df, strategy)
    #wasted_edge_weight_heat(df, strategy)
    
    #walk_success_rate_heat(df, strategy)
    
    #mean_journey_heat(df, strategy)
    #med_journey_heat(df, strategy)
    #cost_heat(df, strategy)
    #walk_entropy_heat(df, strategy)
    
    #cost_pruning_heat(df, strategy)
    #connect_time_heat(df, strategy)
    
    path_pruning_heat(df, strategy)
    chosen_path_entropy_heat(df, strategy)
    
    walk_pruning_heat(df, strategy)
    #chosen_walk_entropy_heat(df, strategy)
    
    #mean_path_len_heat(df, strategy)
    #var_path_len_heat(df, strategy)
    #std_path_len_heat(df, strategy)
   
if __name__ == '__main__':
    main() 
