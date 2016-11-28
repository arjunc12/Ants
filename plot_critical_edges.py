import pylab
import pandas as pd

df = pd.read_csv('critical_edges.csv', header=False, names=['step', 'w1', 'w2'])

df = df.groupby('step', as_index=False).agg(pylab.mean)

pylab.plot(df['step'], df['w1'], c='b')
pylab.plot(df['step'], df['w2'], c='r')
pylab.plot(df['step'], df['w1'] + df['w2'], c='m')
pylab.savefig('critical_edges.png', format='png')