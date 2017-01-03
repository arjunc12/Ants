import os
import argparse
from sys import argv

steps_label = '1k'
doritos_dir = 'achandrasekhar@doritos.snl.salk.edu:/home/achandrasekhar/Documents'
plot_script = 'plot_ant_repair.py'

parser = argparse.ArgumentParser()
parser.add_argument('-s', nargs='+', dest='strategies', required=True)
parser.add_argument('-g', nargs='+', dest='graphs', required=True)
parser.add_argument('-d', nargs='+', dest='decays', required=True)
parser.add_argument('-n', dest='nsteps', type=int, required=True)
parser.add_argument('-l', dest='steps_label', required=True)
parser.add_argument('--sandbox', action='store_true')

args = parser.parse_args()
strategies = args.strategies
graphs = args.graphs
decays = args.decays
nsteps = args.nsteps
steps_label = args.steps_label
sandbox = args.sandbox

for strategy in strategies:
    for graph in graphs:
        for decay in decays:
            prog_name = 'repair'
            if sandbox:
                prog_name += '_sandbox'
            descriptor = '%s_%s_%s_%s' % (prog_name, strategy, graph, decay)
            fname1 = 'ant_%s.csv' % descriptor
            fname2 = 'ant_%s%d.csv' % (descriptor, nsteps)
            scp_command = 'scp %s/%s %s' % (doritos_dir, fname2, fname1)
            print scp_command
            os.system(scp_command)
            plot_label = '%s%s' % (descriptor, steps_label)
            plot_command = 'python %s %s %s' % (plot_script, fname1, plot_label)
            print plot_command
            os.system(plot_command)
            move_command = 'mv %s %s' % (fname1, fname2)
            print move_command
            os.system(move_command)