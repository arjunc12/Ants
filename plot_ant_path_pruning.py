import pandas as pd
from sys import argv
import pylab

def pruning_plot(df, strategy):
    for name1, group1 in df.groupby(['explore', 'decay']):
        explore, decay = name1
        x = []
        y = []
        for name2, group2 in group1.groupby('time'):
            x.append(name2)
            y.append(pylab.nanmean(group2['path_pruning']))
        pylab.figure()
        pylab.plot(x, y)
        pylab.xlabel('time')
        pylab.ylabel('average chosen path entropy')
        figname = 'pruning/path_pruning_%s_e%fd%f.png' % (strategy, explore, decay)
        pylab.savefig(figname, format='png')
        pylab.close()

if __name__ == '__main__':
    filename = argv[1]
    strategy = argv[2]
    
    columns = ['ants', 'explore', 'decay', 'time', 'path_pruning']
    
    df = pd.read_csv(filename, header=None, names = columns, na_values='nan', skipinitialspace=True)
    
    #print max(df['time'])
    
    pruning_plot(df, strategy)