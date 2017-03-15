import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()

columns = ['ants', 'explore', 'decay', 'has_path', 'cost', 'path_entropy', 'walk_entropy', \
               'mean_journey_time', 'median_journey_time', 'walk_success_rate', 'pruning',\
               'connect_time', 'path_pruning', 'chosen_path_entropy', 'walk_pruning', \
               'chosen_walk_entropy', 'wasted_edge_count', 'wasted_edge_weight', 'mean_path_len']

parser.add_argument('-e', '--explores', type=float, required=True, nargs='+')
parser.add_argument('-d', '--decays', type=float, required=True, nargs='+')
parser.add_argument('-me', '--metrics', required=True, nargs='+', choices=columns)
parser.add_argument('-s', '--strategies', required=True, nargs='+')
parser.add_argument('-g', '--graphs', required=True, nargs='+')
parser.add_argument('-dt', '--decay_types', required=True, nargs='+')
parser.add_argument('-m', '--max_steps', type=int, required=True, nargs='+')
parser.add_argument('-v', '--var', action='store_true')
parser.add_argument('-b', '--backtrack', action='store_true')
parser.add_argument('-o', '--one_way', action='store_true')

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

for strategy in strategies:
    for graph in graphs:
        for decay_type in decay_types:
            for steps in max_steps:
                out_items = ['repair', strategy, graph_name, decay_type]
                if backtrack:
                    out_items.append('backtrack')
                if one_way:
                    out_items.append('one_way')
                out_str = '_'.join(out_items)
                fname = 'ant_%s%d.csv' % (out_str, max_steps)
                print fname
                df = pd.read_csv(fname, header=None, names = columns, skipinitialspace=True)
                df = df[['explore', 'decay'] + metrics]
                agg_func = np.nanmean
                if var:
                    agg_func = lambda x : np.nanstd(x, ddof=1)
                df = df.groupby(['explore', 'decay'], as_index=False).agg(agg_func)
                for explore in explores:
                    for decay in decays:
                        df2 = df[(df['explore'] == explore) & (df['decay'] == decay)]
                        print df2
                
                        
                            
                
