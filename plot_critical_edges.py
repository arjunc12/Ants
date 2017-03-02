import pylab
import pandas as pd
from sys import argv

def plot_edge_weights(df, limit):
    df = df.groupby('step', as_index=False).agg(pylab.mean)
    
    pylab.figure()
    pylab.plot(df['step'], df['w1'], c='b', label='dead end')
    pylab.plot(df['step'], df['w2'], c='r', label='path')
    pylab.plot(df['step'], df['w1'] + df['w2'], c='m', label='total')
    pylab.legend()
    title_str = 'critical_edges%d' % limit
    pylab.savefig('%s.png' % title_str, format='png')
    pylab.close()

df = pd.read_csv('critical_edges.csv', header=False, names=['step', 'w1', 'w2'])

max_steps = []
if len(argv) > 1:
    max_steps = map(int, argv[1:])
else:
    max_steps.append(max(df['step']))

max_steps = reversed(sorted(max_steps))

for limit in max_steps:
    df = df[df['step'] <= limit]
    plot_edge_weights(df, limit)