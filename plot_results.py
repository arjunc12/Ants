import os
import argparse
from sys import argv

steps_label = '1k'
plot_script = 'plot_ant_repair.py'

parser = argparse.ArgumentParser()
parser.add_argument('-s', nargs='+', dest='strategies', required=True)
parser.add_argument('-g', nargs='+', dest='graphs', required=True)
parser.add_argument('-dt', nargs='+', dest='decay_types', required=True)
parser.add_argument('-m', dest='max_steps', type=int, required=True)
parser.add_argument('-l', dest='steps_label', required=True)
parser.add_argument('--sandbox', action='store_true')
parser.add_argument('-c', '--supercomputer', default='doritos', dest='supercomputer')
parser.add_argument('-b', '--backtrack', action='store_true', dest='backtrack')
parser.add_argument('-o', '--one_way', action='store_true', dest='one_way')
parser.add_argument('-r', '--remote', action='store_true', dest='remote')

args = parser.parse_args()
strategies = args.strategies
graphs = args.graphs
decay_types = args.decay_types
max_steps = args.max_steps
steps_label = args.steps_label
sandbox = args.sandbox
supercomputer = args.supercomputer
backtrack = args.backtrack
one_way = args.one_way
remote = args.remote

supercomp_dir = 'achandrasekhar@%s.snl.salk.edu:/home/achandrasekhar/Documents' % supercomputer

for strategy in strategies:
    for graph in graphs:
        for decay in decay_types:
            prog_name = 'repair'
            if sandbox:
                prog_name += '_sandbox'
            descriptor_items = [prog_name, strategy, graph, decay]
            if backtrack:
                descriptor_items.append('backtrack')
            if one_way:
                descriptor_items.append('one_way')
            descriptor = '_'.join(descriptor_items)
            #descriptor = '%s_%s_%s_%s' % (prog_name, strategy, graph, decay)
            fname1 = 'ant_%s.csv' % descriptor
            fname2 = 'ant_%s%d.csv' % (descriptor, max_steps)
            if not remote:
                scp_command = 'scp %s/%s %s' % (supercomp_dir, fname2, fname1)
                print scp_command
                os.system(scp_command)
            
            plot_fname = fname1
            if remote:
                plot_fname = fname2
            plot_label = '%s%s' % (descriptor, steps_label)
            plot_command = 'python %s %s %s' % (plot_script, plot_fname, plot_label)
            print plot_command
            os.system(plot_command)
            
            if not remote:
                move_command = 'mv %s %s' % (fname1, fname2)
                print move_command
                os.system(move_command)
