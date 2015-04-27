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
    '''
    x = df['add'].unique()
    y = df['decay'].unique()
    z = pylab.zeros((len(y), len(x)))
    grouped = df.groupby(['add', 'decay'])
    pos = 0
    for name, group in grouped:
        mu = pylab.mean(group['length'])
        i, j = pos % len(y), pos / len(y)
        z[i, j] = mu
        pos += 1
    assert pos == len(x) * len(y)
    pylab.figure()
    hm = pylab.pcolormesh(z, cmap='Reds')
    cb = pylab.colorbar(hm)
    cb.ax.set_ylabel("average walk length")
    #ax.xaxis.set_tick_params(labeltop='on')
    #ax.yaxis.set_tick_params(
    ax = pylab.gca()
    ax.tick_params(top = True, labeltop = True, right=True, labelright = True)
    pylab.xticks(pylab.arange(len(x)) + 0.5, x)
    pylab.yticks(pylab.arange(len(y)) + 0.5, y)
    pylab.xlabel("pheromone add")
    pylab.ylabel("pheromone decay")
    pylab.savefig("ant_walks.png", format="png")
    '''
    def mean_len(group):
        return pylab.mean(group['length'])
    
    heat(df, mean_len, "ant_walks", strategy, "average walk length")

def walk_med_heat(df):
    x = df['add'].unique()
    y = df['decay'].unique()
    z = pylab.zeros((len(y), len(x)))
    grouped = df.groupby(['add', 'decay'])
    pos = 0
    for name, group in grouped:
        mu = pylab.median(group['length'])
        i, j = pos % len(y), pos / len(y)
        z[i, j] = mu
        pos += 1
    assert pos == len(x) * len(y)
    pylab.figure()
    hm = pylab.pcolormesh(z, cmap='Reds')
    cb = pylab.colorbar(hm)
    cb.ax.set_ylabel("median walk length")
    #ax.xaxis.set_tick_params(labeltop='on')
    #ax.yaxis.set_tick_params(
    ax = pylab.gca()
    ax.tick_params(top = True, labeltop = True, right=True, labelright = True)
    pylab.xticks(pylab.arange(len(x)) + 0.5, x)
    pylab.yticks(pylab.arange(len(y)) + 0.5, y)
    pylab.xlabel("pheromone add")
    pylab.ylabel("pheromone decay")
    pylab.savefig("ant_med_walks.png", format="png")
    
def walk_var_heat(df):
    x = df['add'].unique()
    y = df['decay'].unique()
    z = pylab.zeros((len(y), len(x)))
    grouped = df.groupby(['add', 'decay'])
    pos = 0
    for name, group in grouped:
        mu = pylab.var(group['length'], ddof=1)
        i, j = pos % len(y), pos / len(y)
        z[i, j] = mu
        pos += 1
    assert pos == len(x) * len(y)
    pylab.figure()
    hm = pylab.pcolormesh(z, cmap='Reds')
    cb = pylab.colorbar(hm)
    cb.ax.set_ylabel("variance of walk length")
    #ax.xaxis.set_tick_params(labeltop='on')
    #ax.yaxis.set_tick_params(
    ax = pylab.gca()
    ax.tick_params(top = True, labeltop = True, right=True, labelright = True)
    pylab.xticks(pylab.arange(len(x)) + 0.5, x)
    pylab.yticks(pylab.arange(len(y)) + 0.5, y)
    pylab.xlabel("pheromone add")
    pylab.ylabel("pheromone decay")
    pylab.savefig("ant_var_walks.png", format="png")
    
def revisits_heat(df):
    x = df['add'].unique()
    y = df['decay'].unique()
    z = pylab.zeros((len(y), len(x)))
    grouped = df.groupby(['add', 'decay'])
    pos = 0
    for name, group in grouped:
        mu = pylab.mean(group['revisits'])
        i, j = pos % len(y), pos / len(y)
        z[i, j] = mu
        pos += 1
    assert pos == len(x) * len(y)
    pylab.figure()
    hm = pylab.pcolormesh(z, cmap='Reds')
    cb = pylab.colorbar(hm)
    cb.ax.set_ylabel("average number of revisits")
    #ax.xaxis.set_tick_params(labeltop='on')
    #ax.yaxis.set_tick_params(
    ax = pylab.gca()
    ax.tick_params(top = True, labeltop = True, right=True, labelright = True)
    pylab.xticks(pylab.arange(len(x)) + 0.5, x)
    pylab.yticks(pylab.arange(len(y)) + 0.5, y)
    pylab.xlabel("pheromone add")
    pylab.ylabel("pheromone decay")
    pylab.savefig("ant_revisits.png", format="png")
    
def first_walks_heat(dframe):
    df = dframe[dframe['first'] == 1]
    x = df['add'].unique()
    y = df['decay'].unique()
    z = pylab.zeros((len(y), len(x)))
    grouped = df.groupby(['add', 'decay'])
    pos = 0
    for name, group in grouped:
        mu = pylab.mean(group['revisits'])
        print name, mu
        i, j = pos % len(y), pos / len(y)
        z[i, j] = mu
        pos += 1
    assert pos == len(x) * len(y)
    pylab.figure()
    hm = pylab.pcolormesh(z, cmap='Reds')
    cb = pylab.colorbar(hm)
    cb.ax.set_ylabel("average walk length (first 10%)")
    #ax.xaxis.set_tick_params(labeltop='on')
    #ax.yaxis.set_tick_params(
    ax = pylab.gca()
    ax.tick_params(top = True, labeltop = True, right=True, labelright = True)
    pylab.xticks(pylab.arange(len(x)) + 0.5, x)
    pylab.yticks(pylab.arange(len(y)) + 0.5, y)
    pylab.xlabel("pheromone add")
    pylab.ylabel("pheromone decay")
    pylab.savefig("first_walks.png", format="png")
    #pylab.show()

def last_walks_heat(dframe):
    df = dframe[dframe['last'] == 1]
    x = df['add'].unique()
    y = df['decay'].unique()
    z = pylab.zeros((len(y), len(x)))
    grouped = df.groupby(['add', 'decay'])
    pos = 0
    for name, group in grouped:
        mu = pylab.mean(group['revisits'])
        print name, mu
        i, j = pos % len(y), pos / len(y)
        z[i, j] = mu
        pos += 1
    assert pos == len(x) * len(y)
    pylab.figure()
    hm = pylab.pcolormesh(z, cmap='Reds')
    cb = pylab.colorbar(hm)
    cb.ax.set_ylabel("average walk length (last 10%)")
    #ax.xaxis.set_tick_params(labeltop='on')
    #ax.yaxis.set_tick_params(
    ax = pylab.gca()
    ax.tick_params(top = True, labeltop = True, right=True, labelright = True)
    pylab.xticks(pylab.arange(len(x)) + 0.5, x)
    pylab.yticks(pylab.arange(len(y)) + 0.5, y)
    pylab.xlabel("pheromone add")
    pylab.ylabel("pheromone decay")
    pylab.savefig("last_walks.png", format="png")
    
def main():
    strategy = argv[1]
    filename = argv[2]  
    df = pd.read_csv(filename, header=None, names = ['ants', 'add', 'decay', 'length', \
                                                           'first', 'last', 'revisits', 'finished'])
    walk_heat(df, strategy)
    #walk_med_heat(df)
    #walk_var_heat(df)
    #first_walks_heat(df)
    #last_walks_heat(df)
    #revisits_heat(df)
   
if __name__ == '__main__':
    main() 