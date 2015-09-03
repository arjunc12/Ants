import pandas as pd
import datetime
import numpy as np
import pylab

df = pd.read_csv('counts.csv', header=None, names=['source', 'dest', 'dt'])
times = []
for name, group in df.groupby('source'):
    group = group.sort('dt')
    group['dt'] = pd.to_datetime(group['dt'])
    group['delta'] = group['dt'] - group['dt'].shift()
    group['deviate'] = group['dest'] != group['dest'].shift()
    group = group[group['deviate']]
    times += list((group['delta'] / np.timedelta64(1, 's')))[1:]

pylab.hist(times)
pylab.show()
print pylab.mean(times)