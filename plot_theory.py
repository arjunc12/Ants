import pandas as pd
import pylab


df = pd.read_csv('theory.csv', names=['explore', 'decay', 'step', 'w1', 'w2'])
for name, group in df.groupby(['explore', 'decay']):
    explore, decay = name
    fname = 'theory_%0.2f_%0.2f.pdf' % (explore, decay)
    df2 = group.groupby('step', as_index=False).agg(pylab.mean)
    pylab.figure()
    pylab.plot(df2['step'], df2['w1'], label='(1, 2)')
    pylab.plot(df2['step'], df2['w2'], label='(1, 3)')
    pylab.xlabel('time')
    pylab.ylabel('edge weight')
    pylab.legend()
    pylab.savefig(fname, format='pdf')
    pylab.close()