import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--strategies', nargs='+', required=True)
parser.add_argument('-g', '--graphs', nargs='+', required=True)
parser.add_argument('-dt', '--decay_types', nargs='+', required=True)
parser.add_argument('-me', '--metrics', nargs='+', required=True)
parser.add_argument('-l', '--labels', nargs='+', required=True)
parser.add_argument('--sandbox', action='store_true')

remote_dir = 'achandrasekhar@doritos.snl.salk.edu:/home/achandrasekhar/Documents'

args = parser.parse_args()
strategies = args.strategies
graphs = args.graphs
decay_types = args.decay_types
metrics = args.metrics
labels = args.labels
sandbox = args.sandbox

for metric in metrics:
    for strategy in strategies:
        for graph in graphs:
            for decay_type in decay_types:
                for label in labels:
                    local_dir = ['figs']
                    if sandbox:
                        local_dir.append('sandbox')
                    local_dir += [metric, strategy, label]
                    local_dir = '/'.join(local_dir)
                    mkdir_command = 'mkdir -p %s' % local_dir
                    print mkdir_command
                    os.system(mkdir_command)
                    prefix_items = [metric, 'repair']
                    if sandbox:
                        prefix_items.append('sandbox')
                    prefix_items += [strategy, graph, decay_type]
                    prefix = '_'.join(prefix_items)
                    fname = '%s%s.pdf' % (prefix, label)
                    scp_command = 'scp %s/%s %s/%s' % (remote_dir, fname, local_dir, fname)
                    print scp_command
                    os.system(scp_command)                    
