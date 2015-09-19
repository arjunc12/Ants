import pandas as pd
import datetime
import numpy as np
import pylab
from sys import argv
from collections import defaultdict

sheet = int(argv[1])
df = pd.read_csv('reformated_counts%d.csv' % sheet, header=None, names=['source', 'dest', 'dt'])
nondev_times = []
dev_times = []
for name, group in df.groupby('source'):
    group = group.sort('dt')
    group['dt'] = pd.to_datetime(group['dt'])
    group['delta'] = group['dt'] - group['dt'].shift()
    group['deviate'] = group['dest'] != group['dest'].shift()
    dev_group = group[group['deviate']]
    nondev_group = group[group['deviate'] == False]
    dev_times += list(dev_group['delta'] / np.timedelta64(1, 's'))[1:]
    nondev_times += list(nondev_group['delta'] / np.timedelta64(1, 's'))

dev_times = map(int, dev_times)
nondev_times = map(int, nondev_times)

if len(dev_times) > 1:
    dev_desc = 'mu = %0.2f, sigma = %0.2f' % (np.mean(dev_times), np.std(dev_times))
    pylab.hist(dev_times, color='b', alpha=0.5, label='deviate\n' + dev_desc)

if len(nondev_times) > 1:
    nondev_desc = 'mu = %0.2f, sigma = %0.2f' % (np.mean(nondev_times), np.std(nondev_times))
    pylab.hist(nondev_times, color='g', alpha=0.5, label='same\n' + nondev_desc)
pylab.legend()
pylab.xlabel('seconds')
pylab.ylabel('count')
pylab.savefig('dev_hist%d.png' % sheet, format='png')
#pylab.show()

dev_counts = defaultdict(int)
nondev_counts = defaultdict(int)

times = list(set(dev_times + nondev_times))

for dev in dev_times:
    dev_counts[dev] += 1
    
for nondev in nondev_times:
    nondev_counts[nondev] += 1
    
probs = []
for time in times:
    dcount = dev_counts[time]
    ndcount = nondev_counts[time]
    dcount = float(dcount)
    prob = dcount / (dcount + ndcount)
    probs.append(prob)

pylab.figure()    
pylab.scatter(times, probs)
pylab.xlabel('time between choice')
pylab.ylabel('deviation probability')
pylab.savefig('dev_prob%d.png' % sheet, format='png')