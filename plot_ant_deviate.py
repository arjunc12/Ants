import pandas as pd
import pylab
from sys import argv
import argparse
DEBUG = False

def heat(df, group_func, title, strategy, cb_label, sequential=True):
    x = df['explore'].unique()
    y = df['decay'].unique()
    z = pylab.zeros((len(y), len(x)))
    grouped = df.groupby(['explore', 'decay'])
    pos = 0
    max_abs = float('-inf')
    for name, group in grouped:
        val = group_func(group)
        i, j = pos % len(y), pos / len(y)
        z[i, j] = val
        pos += 1
        max_abs = max(max_abs, abs(val))
    assert pos == len(x) * len(y)
    pylab.figure()
    map = 'Reds'
    vmin = 0
    vmax = max_abs
    if not sequential:
        map = 'coolwarm'
        vmin = -max_abs
    #print title, vmin, vmax, map
    hm = pylab.pcolormesh(z, cmap=map, clim=(vmin, vmax))
    cb = pylab.colorbar(hm)
    cb.set_clim(vmin, vmax)
    cb.ax.set_ylabel(cb_label)
    cb.ax.set_ylim(bottom=0)
    pylab.tick_params(which='both', bottom='off', top='off', left='off', right='off', \
    labeltop='off', labelbottom='off', labelleft='off', labelright='off')
    pylab.xlabel("explore probability (%0.2f - %0.2f)" % (min(x), max(x)))
    pylab.ylabel("pheromone decay (%0.2f-%0.2f)" % (min(y), max(y)))
    pylab.savefig("%s_%s.png" % (title, strategy), format="png", transparent=True, bbox_inches='tight')

def describe_group(group):
    print pylab.mean(group['explore']), pylab.mean(group['decay'])

def walk_heat(df, strategy):
    def mean_len(group):
        return pylab.nanmean(group['length'])
    
    heat(df, mean_len, "ant_mean_walks", strategy, "nanmean walk length")

def walk_med_heat(df, strategy):
    def med_len(group):
        return pylab.median(group['length'])
        
    heat(df, med_len, "ant_med_walks", strategy, "median walk length")
    
def walk_var_heat(df, strategy):
    def var_len(group):
        return pylab.var(group['length'], ddof=1)
        
    heat(df, var_len, "ant_var_walks", strategy, "variance of walk length")
    
def revisits_heat(df, strategy):
    def mean_revisits(group):
        return pylab.nanmean(group['revisits'])
        
    heat(df, mean_revisits, "ant_revisits", strategy, "nanmean revisits")
    
def first_walks_heat(df, strategy):
    def first_mean(group):
        g = group[group['first'] == 1]
        return pylab.nanmean(g['length'])
        
    heat(df, first_mean, "first_walks", strategy, "average walk length (first 10%)")

def last_walks_heat(df, strategy):
    def last_mean(group):
        g = group[group['last'] == 1]
        return pylab.nanmean(g['length'])
        
    heat(df, last_mean, "last_walks", strategy, "average walk length (last 10%)")
    
def right_prop_heat(df, strategy):
    def right_prop(group):
        hits = group['hits']
        return float(pylab.count_nonzero(hits)) / len(hits)
    
    heat(df, right_prop, "right_nest_percent", strategy, "percentage of ants to find right nest")
    
def wrong_prop_heat(df, strategy):
    def wrong_prop(group):
        misses = group['misses']
        return float(pylab.count_nonzero(misses)) / len(misses)
    
    heat(df, wrong_prop, "wrong_nest_percent", strategy, "percentage of ants to find wrong nest")
    
def hit_count_heat(df, strategy):
    def hit_count(group):
        return pylab.nanmean(group['hits'])
        
    heat(df, hit_count, "hit_count", strategy, "average times ants found destination nest")
    
def miss_count_heat(df, strategy):
    def miss_count(group):
        return pylab.nanmean(group['misses'])
        
    heat(df, miss_count, "miss_count", strategy, "average times ants returned to origin nest")
    
def success_rate_heat(df, strategy):
    def success_rate(group):
        rate =  sum(group['hits']) / float(sum(group['attempts']))
        return rate
        
    heat(df, success_rate, "success_rate", strategy, "percent of successful attempts")
    
def failure_rate_heat(df, strategy):
    def failure_rate(group):
        rate =  sum(group['misses']) / float(sum(group['attempts']))
        return rate
        
    heat(df, failure_rate, "failure_rate", strategy, "percent of unsuccessful attempts")
    
def success_average_heat(df, strategy):
    def success_average(group):
        return pylab.nanmean(group['hits'] * group['mean_len'])
        
    heat(df, success_average, "success_average", strategy, "average length of successful paths")
    
def success_ratio_heat(df, strategy):
    def success_ratio(group):
        return pylab.nanmean(group['hits']) / pylab.nanmean(group['misses'])
        
    heat(df, success_ratio, "success_ratio", strategy, "ratio of average average successes to average failures")
    
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
            print "path entropy"
            describe_group(group)
        return pylab.nanmean(group['path_entropy'])
        
    heat(df, path_entropy, 'path_entropy', strategy, 'entropy over all possible paths')
    
def min_entropy_dist_heat(df, strategy):
    def min_entropy_dist(group):
        return pylab.nanmean(group['min_etr_dist'])
        
    heat(df, min_entropy_dist, 'min_entropy_dist', strategy, 'length of minimum entropy path')
    
def mean_journey_heat(df, strategy):
    def mean_journey_time(group):
        return pylab.nanmean(group['mean_journey_time']) - 223.13
        
    heat(df, mean_journey_time, 'mean_journey_time', strategy, 'average time for new ants to find nest', sequential=False)
    
def med_journey_heat(df, strategy):
    def med_journey_time(group):
        return pylab.nanmedian(group['mean_journey_time']) - 164
        
    heat(df, med_journey_time, 'median_journey_time', strategy, 'median time for new ants to find nest', sequential=False)
    
def popular_len_heat(df, strategy):
    def mean_popular_len(group):
        return pylab.nanmean(group['popular_len'])
        
    heat(df, mean_popular_len, 'mean_popular_len', strategy, 'average length of most popular path')
    
def walk_entropy_heat(df, strategy):
    def mean_walk_entropy(group):
        if all(pylab.isnan(group['walk_entropy'])):
            print "walk entropy"
            describe_group(group)
        return pylab.nanmean(group['walk_entropy']) - 14.47
        
    heat(df, mean_walk_entropy, 'walk_entropy', strategy, 'entropy over all chosen walks', sequential=False)
    
def path_success_rate_heat(df, strategy):
    def path_success_rate(group):
        success_rate = pylab.nansum(group['has_path']) / float(pylab.count_nonzero(~pylab.isnan(group['has_path'])))
        if success_rate == 0:
            print "path success rate"
            describe_group(group)
        return success_rate
    heat(df, path_success_rate, 'path_success_rate', strategy, 'proportion of times ants successfully created a path')
    
def walk_success_rate_heat(df, strategy):
    def walk_success_rate(group):
        success_rate = pylab.nanmean(group['walk_success_rate'])
        if success_rate == 0:
            print "walk success rate"
            describe_group(group)
        return success_rate - 0.9933417141012928
    heat(df, walk_success_rate, 'walk_success_rate', strategy, 'proportion successful walks for new ants', sequential=False)

def pruning_heat(df, strategy):
    def pruning(group):
        return pylab.nanmean(group['pruning'])
        
    heat(df, pruning, 'pruning', strategy, 'average edge pruning done between start and end')
    
def path_pruning_heat(df, strategy):
    def path_pruning(group):
        if all(pylab.isnan(group['path_pruning'])):
            print "path pruning"
            print group
            
        return pylab.nanmean(group['path_pruning'])
            
    heat(df, path_pruning, 'path_pruning', strategy, 'average path pruning done between start and end')
    
def main():
    filename = argv[1]
    strategy = argv[2]
    
    columns = ['ants', 'explore', 'decay', 'has_path', 'cost', 'path_entropy', 'walk_entropy', \
               'mean_journey_time', 'median_journey_time', 'walk_success_rate', 'pruning',\
               'connect_time', 'path_pruning']
    
    df = pd.read_csv(filename, header=None, names = columns, na_values='nan', skipinitialspace=True)
    
    if DEBUG:
        print df['walk_success_rate']
        print pylab.nanmax(df['walk_success_rate'])
        return None
    
    path_success_rate_heat(df, strategy)
    walk_success_rate_heat(df, strategy)
    
    mean_journey_heat(df, strategy)
    med_journey_heat(df, strategy)
    path_entropy_heat(df, strategy)
    cost_heat(df, strategy)
    walk_entropy_heat(df, strategy)
    
    pruning_heat(df, strategy)
    connect_time_heat(df, strategy)
    
    path_pruning_heat(df, strategy)
   
if __name__ == '__main__':
    main() 