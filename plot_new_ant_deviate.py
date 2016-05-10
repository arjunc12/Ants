import pandas as pd
import pylab
from sys import argv
import argparse
DEBUG = False

def heat(df, group_func, title, strategy, cb_label, map='Reds'):
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
    hm = pylab.pcolormesh(z, cmap=map)
    cb = pylab.colorbar(hm)
    cb.ax.set_ylabel(cb_label)
    pylab.tick_params(which='both', bottom='off', top='off', left='off', right='off', \
    labeltop='off', labelbottom='off', labelleft='off', labelright='off')
    pylab.xlabel("explore probability (%0.2f - %0.2f)" % (min(x), max(x)))
    pylab.ylabel("pheromone decay (%0.2f-%0.2f)" % (min(y), max(y)))
    pylab.savefig("%s_%s.png" % (title, strategy), format="png")
    
def mean_journey_time_heat(df, strategy):
    def mean_journey_time(group):
        return pylab.nanmean(group['journey_time'])
    heat(df, mean_journey_time, 'mean_journey_time', strategy, 'mean time for new ants to reach nest')
    
def med_journey_time_heat(df, strategy):
    def med_journey_time(group):
        return pylab.nanmedian(group['journey_time'])
    heat(df, med_journey_time, 'median_journey_time', strategy, 'median time for new ants to reach nest')
    
def logmean_journey_time_heat(df, strategy):
    def logmean_journey_time(group):
        return pylab.log10(pylab.nanmean(group['journey_time']))
    heat(df, logmean_journey_time, 'logmean_journey_time', strategy, 'log of mean time for new ants to reach nest')
    
def logmed_journey_time_heat(df, strategy):
    def logmed_journey_time(group):
        return pylab.log10(pylab.nanmedian(group['journey_time']))
    heat(df, logmed_journey_time, 'logmedian_journey_time', strategy, 'log of median time for new ants to reach nest')

def walk_success_rate_heat(df, strategy):
    def walk_success_rate(group):
        group2 = group[group['journey_time'] != -1] 
        return float(len(group2.index)) / float(len(group.index))
    heat(df, walk_success_rate, 'walk_success_rate', strategy, 'proportion of time new ants successfully make a path')
    
def main():
    filename = argv[1]
    strategy = argv[2]
    columns = ['explore', 'decay', 'journey_time'] 
    df = pd.read_csv(filename, header=None, names = columns, skipinitialspace=True)
    df2 = df[df['journey_time'] != -1]
    print max(df['journey_time'])
    mean_journey_time_heat(df2, strategy)
    med_journey_time_heat(df2, strategy)
    logmean_journey_time_heat(df2, strategy)
    logmed_journey_time_heat(df2, strategy)
    walk_success_rate_heat(df, strategy)
    
   
if __name__ == '__main__':
    main() 