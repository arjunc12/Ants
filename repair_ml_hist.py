import pandas as pd
import pylab
import numpy as np
import argparse

def hist_values(parameter, group, strategy, decay_type, label, y_limit=None):
    values = group[parameter]
    values = list(values)
    
    binsize = 0.05
    if 'zoom' in label:
        binsize = 0.01
    
    if y_limit == None:
        y_limit = max(values)
    cutoff = 1
    #weights = np.ones_like(values)/float(len(values))
    
    weights = group['num_lines'] / sum(group['num_lines'])
    weights = np.array(weights)
    
    mu = pylab.average(values, weights=weights)
    sigma2 = pylab.var(values)
    
    pylab.figure()
    pylab.hist(values, weights=weights, bins=np.arange(0, cutoff + binsize, binsize))
    title_items = []
    title_items.append('%s maximum likelihood values %s %s %s' % (parameter, strategy, decay_type, label))
    title_items.append('mean of estimates = %f' % mu)
    title_items.append('variance of estimates = %f' % sigma2)
    title_str = '\n'.join(title_items)
    #pylab.title(parameter + ' maximum likelihood values ' + str(strategy) + ' ' + str(outname))
    #pylab.title(title_str)
    print title_str
    pylab.xlabel('%s mle' % parameter, fontsize=20)
    pylab.ylabel('weighted proportion', fontsize=20)
    pylab.xlim((0, 1))
    pylab.ylim((0, y_limit))
    pylab.savefig('repair_ml_hist_%s_%s_%s_%s.pdf' % (parameter, strategy, decay_type, label), format='pdf')
    pylab.close()
 
def main():   
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--strategies', default=None, nargs='+')
    parser.add_argument('-dt', '--decay_types', default=None, nargs='+')
    parser.add_argument('-l', '--labels', default=None, nargs='+')
    parser.add_argument('-p', '--parameters', default=None, nargs='+')
    parser.add_argument('-ylim', '--y_limit', default=None, type=float)
    
    args = parser.parse_args()
    strategies = args.strategies
    decay_types = args.decay_types
    labels = args.labels
    parameters = args.parameters
    y_limit = args.y_limit
    
    columns = ['explore', 'decay', 'strategy', 'decay_type', 'label', 'num_lines']
    ml_values = pd.read_csv('repair_ml_hist.csv',  header=None, names = columns, na_values='nan',\
                            skipinitialspace=True)
    for name, group in ml_values.groupby(['strategy', 'decay_type', 'label']):
        strategy, decay_type, label = name
        if strategies != None and strategy not in strategies:
            continue
        if decay_types != None and decay_type not in decay_types:
            continue
        if labels != None and label not in labels:
            continue
        print name
        for parameter in ['explore', 'decay']:
            if (parameters == None) or (parameters != None and parameter in parameters):
                hist_values(parameter, group, strategy, decay_type, label, y_limit)
        
        #hist_values('explore', group, strategy, decay_type, label)
        #hist_values('decay', group, strategy, decay_type, label)
        
if __name__ == '__main__':
    main()