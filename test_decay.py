import pandas as pd
import datetime
import numpy as np
import pylab
from sys import argv
from collections import defaultdict
from scipy.stats import ttest_1samp
from interval_utils import get_times, get_all_times

def hist_times(name, times):
    min_time, max_time = min(times), max(times)
    times = filter(lambda x : x <= 200, times)
    N = len(times)
    
    binsize = 20
    weights = np.ones_like(times)/float(len(times))
    pylab.hist(times, weights=weights)#, bins=np.arange(0, max(times) + binsize, binsize))
    pylab.ylim(0, 1)
    pylab.xlabel('time (seconds)')
    pylab.ylabel('proportion of times in that time range')
    pylab.title('times between traversals of a unique edge\nN = %d, min = %0.2f, max = %0.2f' % (N, min_time, max_time))
    pylab.savefig('interval_counts_%s.png' % name, format='png', bbox_inches='tight')
    #print "show"
    #pylab.show()
    
def cumulative_plot_times(name, sheets):
    times = get_all_times(sheets)
    hist_times(name, times)
    
def plot_time(sheet):
    hist_times(sheet, times)
    
def plot_times(sheets):
    for sheet in sheets:
        plot_time(sheet)

def test_decay(decay, sheets):
    mu = 1.0 / decay
    times = get_all_times(sheets)
    return ttest_1samp(times, mu)
    
        
if __name__ == '__main__':
    #decay = float(argv[1])
    name = argv[1]
    sheets = argv[2:]
    #plot_times(sheets)
    cumulative_plot_times(name, sheets)
    #print test_decay(decay, sheets)
    
    