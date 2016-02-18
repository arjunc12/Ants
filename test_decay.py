import pandas as pd
import datetime
import numpy as np
import pylab
from sys import argv
from collections import defaultdict
from scipy.stats import ttest_1samp

def get_times(sheet):
    df = pd.read_csv('reformated_counts%d.csv' % sheet, header=None, names=['source', 'dest', 'dt'])
    def sort_edge(row):
        return '-'.join(sorted(row['source'].strip() + row['dest'].strip()))
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

def test_decay(decay, sheets):
    mu = 1.0 / decay
    times = get_all_times(sheets)
    return ttest_1samp(times, mu)
    
        
if __name__ == '__main__':
    decay = float(argv[1])
    sheets = map(int, argv[2:])
    print test_decay(decay, sheets)
    
    