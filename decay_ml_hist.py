import pandas as pd
import pylab
import numpy as np

def hist_values(parameter, values, strategy, outname):
    values = list(values)
    binsize = 0.05
    cutoff = 1
    weights = np.ones_like(values)/float(len(values))
    pylab.figure()
    pylab.hist(values, weights=weights, bins=np.arange(0, cutoff + binsize, binsize))
    pylab.title(parameter + ' maximum likelihood values ' + str(strategy) + ' ' + str(outname))
    pylab.savefig('decay_ml_hist_%s_%s_%s.png' % (parameter, strategy, outname), format='png')
    
columns = ['explore', 'decay', 'strategy', 'outname']
ml_values = pd.read_csv('decay_ml.csv',  header=None, names = columns, na_values='nan',\
                        skipinitialspace=True)
for name, group in ml_values.groupby(['strategy', 'outname']):
    strategy, outname = name
    hist_values('explore', group['explore'], strategy, outname)
    hist_values('decay', group['decay'], strategy, outname)