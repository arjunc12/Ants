import pandas as pd
import datetime
import numpy as np
import pylab
from sys import argv
from collections import defaultdict
from scipy.stats import ttest_1samp

def get_times(sheet):
    print sheet
    df = pd.read_csv('reformated_counts%s.csv' % sheet, header=None, names=['source', 'dest', 'dt'])
    def sort_edge(row):
        return '-'.join(sorted(str(row['source']).strip() + str(row['dest']).strip()))
    df['lex_edge'] = df.apply(sort_edge, axis=1)
    times = []
    for name, group in df.groupby('lex_edge'):
        group = group.sort('dt')
        group['dt'] = pd.to_datetime(group['dt'])
        group['delta'] = group['dt'] - group['dt'].shift()
        times += list((group['delta'] /  np.timedelta64(1, 's'))[1:])
    return times

def get_all_times(sheets):
    times = []
    for sheet in sheets:
        times += get_times(sheet)
    return times

def plot_times(name, sheets):
    times = get_all_times(sheets)
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
    

def test_decay(decay, sheets):
    mu = 1.0 / decay
    times = get_all_times(sheets)
    return ttest_1samp(times, mu)
    
        
if __name__ == '__main__':
    #decay = float(argv[1])
    name = argv[1]
    sheets = argv[2:]
    plot_times(name, sheets)
    #print test_decay(decay, sheets)
    
    