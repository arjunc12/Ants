import pandas as pd
import pylab
from sys import argv
import argparse

def heat(df, group_func, title, strategy, cb_label):
    x = df['add'].unique()
    y = df['decay'].unique()
    z = pylab.zeros((len(y), len(x)))
    grouped = df.groupby(['add', 'decay'])
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
    ax = pylab.gca()
    ax.tick_params(top = True, labeltop = True, right=True, labelright = True)
    pylab.xticks(pylab.arange(len(x)) + 0.5, x)
    pylab.yticks(pylab.arange(len(y)) + 0.5, y)
    pylab.xlabel("pheromone add")
    pylab.ylabel("pheromone decay")
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
    
def main():
    strategy = argv[1]
    filename = argv[2]  
    df = pd.read_csv(filename, header=None, names = ['ants', 'add', 'decay', 'length', \
                                                           'first', 'last', 'revisits', 'finished'])
    walk_heat(df, strategy)
    walk_med_heat(df, strategy)
    walk_var_heat(df, strategy)
    first_walks_heat(df, strategy)
    last_walks_heat(df, strategy)
    revisits_heat(df, strategy)
   
if __name__ == '__main__':
    main() 