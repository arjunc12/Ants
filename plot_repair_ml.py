import matplotlib as mpl
mpl.use('agg')
import pandas as pd
import argparse
import seaborn as sns
import numpy as np
import pylab

COLUMNS = ['sheet', 'num_lines', 'strategy', 'decay_type', 'ghost', 'explore', 'decay',\
           'likelihood']

NOCUT = ['6', '7a', '7c', '8a', '8b', '9b', '14','15', '16', '17', '21a', '21b', '23a', '23d']
CUT = ['7b', '9a', '9c', '9d', '10', '11', '12', '13', '18', '19', '20', '21c', '22', '23b', '23c', '23e', '23f']
ALL = NOCUT + CUT

NOCUT2 = ['6', '7a', '7c', '9b', '14','15', '16', '21a', '21b', '23a', '23d']
ALL2 = NOCUT2 + CUT

DATASETS_DIR = 'datasets/reformated_csv'
OUT_DIR = 'ml_plots'
ML_OUTFILE = '%s/repair_ml.csv' % OUT_DIR

def print_ml_parameters(df, strategies, decay_types):
    df2 = df[(df['strategy'].isin(strategies)) & (df['decay_type'].isin(decay_types))]
    for name, group in df2.groupby(['strategy', 'decay_type']):
        strategy, decay_type = name
        print "%s, %s" % (strategy, decay_type)
        max_likelihood = float('-inf')
        max_likelihood_params = []
        for name2, group2 in group.groupby(['explore', 'decay']):
            explore, decay = name2
            likelihood = sum(group2['likelihood'])
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                max_likelihood_params = [(explore, decay)]
            elif likelihood == max_likelihood:
                max_likelihood_params.append((explore, decay))
        print "maximum likelihood %f" % max_likelihood
        for explore, decay in max_likelihood_params:
            print "e = %f, d=%f" % (explore, decay)

def ml_heat(df, strategies, decay_types):
    df2 = df[(df['strategy'].isin(strategies)) & (df['decay_type'].isin(decay_types))]
    for name, group in df2.groupby(['strategy', 'decay_type']):
        group = group.groupby(['explore', 'decay'], as_index=False).agg({'likelihood' : np.sum})
        group = group.pivot('decay', 'explore', 'likelihood')
        pylab.figure()
        ax = sns.heatmap(group, cmap='nipy_spectral', xticklabels=False, yticklabels=False, linewidths=0)
        ax.invert_yaxis()
        pylab.savefig('sandbox.pdf')
        pylab.close()

def ml_plot(df, strategies, decay_types):
    df2 = df[(df['strategy'].isin(strategies)) & (df['decay_type'].isin(decay_types))]
    for name, group in df2.groupby(['strategy', 'decay_type']):
        x = []
        y = []
        for name2, group2 in group.groupby('decay'):
            decay = name2
            group2 = group2.groupby('explore', as_index=False).agg({'likelihood' : np.sum})
            max_likelihood = group2['likelihood'].max()
            group2 = group2[group2['likelihood'] == max_likelihood]
            best_explores = group2['explore']
            for explore in best_explores:
                x.append(decay)
                y.append(explore)
        pylab.figure()
        pylab.scatter(x, y)
        pylab.savefig('figs/sandbox.pdf')
        pylab.close()

def print_variance(df, strategies, decay_types):
    df2 = df[(df['strategy'].isin(strategies)) & (df['decay_type'].isin(decay_types))]
    for name, group in df2.groupby(['strategy', 'decay_type']):
        print name
        max_explores = []
        max_decays = []
        for sheet in group['sheet'].unique():
            group2 = group[group['sheet'] == sheet]
            max_likelihood = max(group2['likelihood'])
            group2 = group2[group2['likelihood'] == max_likelihood]
            #print group2
            explores = list(group2['explore'])
            decays = list(group2['decay'])
            max_explores += explores
            max_decays += decays
        print "explore", pylab.var(max_explores, ddof=1)
        print "decay", pylab.var(max_decays, ddof=1)

def filter_df(df, emin, emax, dmin, dmax, ghost, sheets):
    '''
    df =  df[(emin <= df['explore']) & (df['explore'] <= emax) & \
             (dmin <= df['decay']) & (df['decay'] <= dmax) & \
             (df['ghost'] == int(ghost)) & (df['sheet'].isin(sheets))]
    '''
    df = df[df['explore'] >= emin]
    df = df[df['explore'] <= emax]
    
    df = df[df['decay'] >= dmin]
    df = df[df['decay'] <= dmax]
    
    df = df[df['ghost'] == int(ghost)]
   
    df = df[df['sheet'].isin(sheets)]

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sh', '--sheets', nargs='+', required=True)
    parser.add_argument('-l', '--label', default=None)
    
    parser.add_argument('-s', '--strategies', nargs='+', required=True)
    parser.add_argument('-dt', '--decay_types', nargs='+', required=True)
    
    parser.add_argument('-emin', type=float, default=0.01)
    parser.add_argument('-emax', type=float, default=0.99)
    
    parser.add_argument('-dmin', type=float, default=0.01)
    parser.add_argument('-dmax', type=float, default=0.99)

    parser.add_argument('-g', '--ghost', action='store_true')
    parser.add_argument('--heat', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--hist', action='store_true')
    parser.add_argument('--print', action='store_true', dest='print_ml')
    parser.add_argument('-v', '--var', action='store_true', dest='print_var')

    args = parser.parse_args()

    sheets = args.sheets
    label = args.label

    if label == None:
        assert len(sheets) == 1
        label = sheets

    if sheets == ['nocut']:
        sheets = NOCUT
    elif sheets == ['nocut2']:
        sheets = NOCUT2
    elif sheets == ['cut']:
        sheets = CUT
    elif sheets == ['all']:
        sheets = ALL
    elif sheets == ['all2']:
        sheets = ALL2
    else:
        assert len(sheets) > 1
        acceptable_sheets = set(ALL)
        for sheet in sheets:
            assert sheet in acceptable_sheets
   
    strategies = args.strategies
    decay_types = args.decay_types

    emin = args.emin
    emax = args.emax

    dmin = args.dmin
    dmax = args.dmax

    ghost = args.ghost
       
    heat = args.heat
    plot = args.plot
    hist = args.hist
    print_ml = args.print_ml
    print_var = args.print_var

    df = pd.read_csv(ML_OUTFILE, names = COLUMNS, skipinitialspace=True)
    df['sheet'] = df['sheet'].astype(str)
    df.drop_duplicates(inplace=True)
    df = filter_df(df, emin, emax, dmin, dmax, ghost, sheets)

    if heat:
        ml_heat(df, strategies, decay_types)
    if plot:
        ml_plot(df, strategies, decay_types)
    if print_ml:
        print_ml_parameters(df, strategies, decay_types)
    if print_var:
        print_variance(df, strategies, decay_types)

if __name__ == '__main__':
    main()
