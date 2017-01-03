import pandas as pd
import pylab
import numpy as np

def hist_values(parameter, values, strategy, outname):
    values = list(values)
    mu = pylab.mean(values)
    sigma2 = pylab.var(values)
    binsize = 0.05
    cutoff = 1
    weights = np.ones_like(values)/float(len(values))
    pylab.figure()
    pylab.hist(values, weights=weights, bins=np.arange(0, cutoff + binsize, binsize))
    title_items = []
    title_items.append('%s maximum likelihood values %s %s' % (parameter, strategy, outname))
    #title_items.append('mean of estimates = %f' % mu)
    title_items.append('variance of estimates = %f' % sigma2)
    title_str = '\n'.join(title_items)
    #pylab.title(parameter + ' maximum likelihood values ' + str(strategy) + ' ' + str(outname))
    pylab.title(title_str)
    pylab.savefig('repair_ml_hist_%s_%s_%s.png' % (parameter, strategy, outname), format='png')
    pylab.close()
    
columns = ['explore', 'decay', 'strategy', 'outname']
ml_values = pd.read_csv('repair_ml.csv',  header=None, names = columns, na_values='nan',\
                        skipinitialspace=True)
for name, group in ml_values.groupby(['strategy', 'outname']):
    strategy, outname = name
    print name
    hist_values('explore', group['explore'], strategy, outname)
    hist_values('decay', group['decay'], strategy, outname)