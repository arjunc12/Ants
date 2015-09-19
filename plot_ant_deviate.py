import pandas as pd
import pylab
from sys import argv
import argparse

def heat(df, group_func, title, strategy, cb_label):
    x = df['explore'].unique()
    y = df['decay'].unique()
    z = pylab.zeros((len(y), len(x)))
    grouped = df.groupby(['explore', 'decay'])
    pos = 0
    for name, group in grouped:
        val = group_func(group)
        i, j = pos % len(y), pos / len(y)
        z[i, j] = val
        pos += 1
    assert pos == len(x) * len(y)
    pylab.figure()
    hm = pylab.pcolormesh(z, cmap='Reds')
    cb = pylab.colorbar(hm)
    cb.ax.set_ylabel(cb_label)
    #ax.xaxis.set_tick_params(labeltop='on')
    #ax.yaxis.set_tick_params(
    #ax = pylab.gca()
    #ax.tick_params(top = True, labeltop = True, right=True, labelright = True)
    #pylab.xticks(pylab.arange(len(x)) + 0.5, sorted(x), rotation=90)
    #pylab.yticks(pylab.arange(len(y)) + 0.5, sorted(y))
    pylab.xlabel("explore probability (%0.2f - %0.2f)" % (min(x), max(x)))
    pylab.ylabel("pheromone decay (%0.2f-%0.2f)" % (min(y), max(y)))
    pylab.savefig("%s_%s.png" % (title, strategy), format="png")

def walk_heat(df, strategy):
    def mean_len(group):
        return pylab.mean(group['length'])
    
    heat(df, mean_len, "ant_mean_walks", strategy, "mean walk length")

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
        return pylab.mean(group['revisits'])
        
    heat(df, mean_revisits, "ant_revisits", strategy, "mean revisits")
    
def first_walks_heat(df, strategy):
    def first_mean(group):
        g = group[group['first'] == 1]
        return pylab.mean(g['length'])
        
    heat(df, first_mean, "first_walks", strategy, "average walk length (first 10%)")

def last_walks_heat(df, strategy):
    def last_mean(group):
        g = group[group['last'] == 1]
        return pylab.mean(g['length'])
        
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
        return pylab.mean(group['hits'])
        
    heat(df, hit_count, "hit_count", strategy, "average times ants found destination nest")
    
def miss_count_heat(df, strategy):
    def miss_count(group):
        return pylab.mean(group['misses'])
        
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
        return pylab.mean(group['hits'] * group['mean_len'])
        
    heat(df, success_average, "success_average", strategy, "average length of successful paths")
    
def success_ratio_heat(df, strategy):
    def success_ratio(group):
        return pylab.mean(group['hits']) / pylab.mean(group['misses'])
        
    heat(df, success_ratio, "success_ratio", strategy, "ratio of average average successes to average failures")
    
def main():
    filename = argv[1]
    strategy = argv[2]
    columns = ['ants', 'explore', 'decay', 'first', 'last', 'revisits', 'hits', 'misses', 'mean_len', 'attempts'] 
    df = pd.read_csv(filename, header=None, names = columns)
    #print df['attempts']
    #walk_heat(df, strategy)
    #walk_med_heat(df, strategy)
    #walk_var_heat(df, strategy)
    #first_walks_heat(df, strategy)
    #last_walks_heat(df, strategy)
    #revisits_heat(df, strategy)
    right_prop_heat(df, strategy)
    wrong_prop_heat(df, strategy)
    hit_count_heat(df, strategy)
    miss_count_heat(df, strategy)
    success_rate_heat(df, strategy)
    failure_rate_heat(df, strategy)
    success_average_heat(df, strategy)
    success_ratio_heat(df, strategy)
   
if __name__ == '__main__':
    main() 