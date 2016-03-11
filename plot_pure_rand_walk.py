import pandas as pd
import pylab
from collections import defaultdict
from scipy.stats import entropy

df = pd.read_csv('pure_random_walk.csv', header=None, names = ['steps', 'path'])
path_counts = defaultdict(int)
N = 0
for path in df['path']:
    N += 1
    path_counts[path] += 1
etr = entropy(path_counts.values())
pylab.hist(df['steps'])
pylab.title("N = %d, mean = %0.2f, median = %d, entropy=%0.2f" % (N, pylab.mean(df['steps']), pylab.median(df['steps']), etr))
pylab.savefig('pure_rand_walk.png', format='png')