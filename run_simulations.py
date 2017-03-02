import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--strategies', nargs='+', required=True, dest='strategies')
parser.add_argument('-g', '--graphs', nargs='+', required=True, dest='graphs')
parser.add_argument('-dt', '--decay_types', nargs='+', required=True, dest='decay_types')
parser.add_argument('-l', '--labels', nargs='+', required=True, dest='labels')
parser.add_argument('--sandbox', action='store_true')
parser.add_argument('-x', '--num_iters', type=int, default=1, dest='num_iters')
parser.add_argument('-b', '--backtrack', action='store_true', dest='backtrack')
parser.add_argument('-o', '--one_way', action='store_true', dest='one_way')

args = parser.parse_args()
strategies = args.strategies
graphs = args.graphs
decay_types = args.decay_types
labels = args.labels
sandbox = args.sandbox
num_iters = args.num_iters
backtrack = args.backtrack
one_way = args.one_way

for i in xrange(num_iters):
    for strategy in strategies:
        for graph in graphs:
            for decay_type in decay_types:
                for label in labels:
                    prog_name = 'ant_repair'
                    if sandbox:
                        prog_name += '_sandbox'
                    descriptor_items = [prog_name, strategy, graph, decay_type]
                    if backtrack:
                        descriptor_items.append('backtrack')
                    if one_way:
                        descriptor_items.append('one_way')
                    #descriptor_items.append(label)
                    descriptor = '_'.join(descriptor_items)
                    #command_str = 'bash %s_%s_%s_%s%s.sh' % (prog_name, strategy, graph, decay_type, label)
                    command_str = 'bash %s%s.sh' % (descriptor, label)
                    print command_str
                    os.system(command_str)
