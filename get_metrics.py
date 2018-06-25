import pandas as pd
import argparse
import numpy as np
from plot_ant_repair import COLUMNS

def nan_std(x):
    return np.std(x, ddof=1)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--explores', type=float, required=True, nargs='+')
    parser.add_argument('-d', '--decays', type=float, required=True, nargs='+')
    parser.add_argument('-me', '--metrics', required=True, nargs='+', choices=COLUMNS)
    parser.add_argument('-s', '--strategies', required=True, nargs='+')
    parser.add_argument('-g', '--graphs', required=True, nargs='+')
    parser.add_argument('-dt', '--decay_types', required=True, nargs='+')
    parser.add_argument('-m', '--max_steps', type=int, required=True, nargs='+')
    parser.add_argument('-v', '--var', action='store_true')
    parser.add_argument('-b', '--backtrack', action='store_true')
    parser.add_argument('-o', '--one_way', action='store_true')
    parser.add_argument('--sandbox', action='store_true')
    parser.add_argument('--future', action='store_true')

    args = parser.parse_args()
    explores = args.explores
    decays = args.decays
    metrics = args.metrics
    strategies = args.strategies
    graphs = args.graphs
    decay_types = args.decay_types
    max_steps = args.max_steps
    var = args.var
    backtrack = args.backtrack
    one_way = args.one_way
    sandbox = args.sandbox
    future = args.future

    for strategy in strategies:
        for graph in graphs:
            for decay_type in decay_types:
                for steps in max_steps:
                    out_items = ['repair']
                    if future:
                        out_items.append('future')
                    if sandbox:
                        out_items.append('sandbox')
                    out_items += [strategy, graph, decay_type]
                    if backtrack:
                        out_items.append('backtrack')
                    if one_way:
                        out_items.append('one_way')
                    out_str = '_'.join(out_items)
                    fname = 'ant_%s%d.csv' % (out_str, steps)
                    print fname
                    df = pd.read_csv(fname, header=None, names = COLUMNS, skipinitialspace=True)
                    df = df[['explore', 'decay'] + metrics]
                    agg_funcs = [np.nanmean]
                    if var:
                        agg_funcs += [nan_std]
                    agg_dict = {}
                    for metric in metrics:
                        agg_dict[metric] = agg_funcs
                    df = df.groupby(['explore', 'decay'], as_index=False).agg(agg_dict)
                    for explore in explores:
                        for decay in decays:
                            df2 = df[(df['explore'] == explore) & (df['decay'] == decay)]
                            print df2

if __name__ == '__main__':
    main()
