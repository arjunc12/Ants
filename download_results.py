import os
import argparse

parser = argparse.Argument_Parser()
parser.add_argument('-s', '--strategies', nargs='+', required=True)
parser.add_argument('-g', '--graphs', nargs='+', required=True)
parser.add_argument('-dt', '--decay_types', nargs='+', required=True)
parser.add_argument('-me', '--metrics', nargs='+', required=True)
parser.add_argument('-l', '--labels', nargs='+', required=True)

remote_dir = 'achandrasekhar@doritos.snl.salk.edu:/home/achandrasekhar/Documents/'

args = parser.parse_args()
strategies = args.strategies
graphs = args.graphs
decay_types = args.decay_types
metrics = args.metrics
labels = args.labels

for metric in metrics:
    for strategy in strategies:
        for graph in graphs:
            for decay_type in decay_types:
                for label in labels:
                    fname = '%s_repair_%s_%s_%s%s.pdf' % (metric, strategy, graph, decay_type, label)
                    scp_command = 'scp %s/%s %s' % (remote_name, fname, fname)
                    
                    
