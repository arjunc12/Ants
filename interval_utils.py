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
    