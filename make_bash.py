'''
Module for making bash files to run scripts on supercomputer.
Automates writing and naming of files, which especially helps with naming the
files correctly 
'''

import argparse

parser = argparse.ArgumentParser()

# required arguments
parser.add_argument('-s', '--strategies', nargs='+', required=True)
parser.add_argument('-g', '--graphs', nargs='+', required=True)
parser.add_argument('-dt', '--decay_types', nargs='+', required=True)
parser.add_argument('-m', '--steps', required=True, type=int)
parser.add_argument('-l', '--steps_label', required=True)

# which problem to solve
parser.add_argument('-p', '--program', default='repair')

# run the sandbox script instead of the main repair script
# sandbox script used for testing new ideas like modulating parameters, new choice functions, etc.
parser.add_argument('--sandbox', action='store_true')

# optional arguments

# explore, decay ranges
parser.add_argument('-emin', type=float, default=0.05)
parser.add_argument('-emax', type=float, default=0.95)
parser.add_argument('-dmin', type=float, default=0.05)
parser.add_argument('-dmax', type=float, default=0.95)

# explore, decay, add increments
parser.add_argument('-estep', type=float, default=0.05)
parser.add_argument('-dstep', type=float, default=0.05)
parser.add_argument('-a', '--add', type=float, default=1)

# number of different processes to use, so number of trials to run for each
# parameter combination
parser.add_argument('-x', '--num_iters', type=int, default=50)

# number of ants in each simulation
parser.add_argument('-n', '--num_ants', type=int, default=100)

# node, edge queue limits
parser.add_argument('-nql', '--node_queue_lim', type=int, default=1, dest='nql')
parser.add_argument('-eql', '--edge_queue_lim', type=int, default=1, dest='eql')

# options for backtracking or performing uni-directional search
parser.add_argument('-b', '--backtrack', action='store_true', dest='backtrack')
parser.add_argument('-o', '--one_way', action='store_true', dest='one_way')

# parse arguments
args = parser.parse_args()
strategies = args.strategies
graphs = args.graphs
decay_types = args.decay_types
steps = args.steps
steps_label = args.steps_label

program = args.program

emin = args.emin
emax = args.emax
dmin = args.dmin
dmax = args.dmax

estep = args.estep
dstep = args.dstep

add = args.add

num_iters = args.num_iters
num_ants = args.num_ants

nql = args.nql
eql = args.eql

backtrack = args.backtrack
one_way = args.one_way

sandbox = args.sandbox

for strategy in strategies:
    for graph in graphs:
        for decay_type in decay_types:
            out_items = [program]
            if sandbox:
                out_items.append('sandbox')
            out_items += [strategy, graph, decay_type]
            if backtrack:
                out_items.append('backtrack')
            if one_way:
                out_items.append('one_way')
            out_str = '_'.join(out_items)
            
            fname = 'ant_%s%s.sh' % (out_str, steps_label)
            print fname
            f = open(fname, 'w')

            # create script variables for all the parameters
            f.write('strategy=\'%s\'\n' % strategy)
            f.write('graph=\'%s\'\n' % graph)
            f.write('decay_type=\'%s\'\n' % decay_type)
            f.write('\n')
            
            f.write('emin=%f\n' % emin)
            f.write('emax=%f\n' % emax)
            f.write('\n')
            
            f.write('dmin=%f\n' % dmin)
            f.write('dmax=%f\n' % dmax)
            f.write('\n')
            
            f.write('add=%f\n' % add)
            f.write('\n')

            f.write('estep=%f\n' % estep)
            f.write('dstep=%f\n' % dstep)
            f.write('\n')

            f.write('x=%d\n' % num_iters)
            f.write('n=%d\n' % num_ants)
            f.write('m=%d\n' % steps)
            f.write('\n')

            f.write('nq=%d\n' % nql)
            f.write('eq=%d\n' % eql)
            
            pyscript = 'ant_' + program
            if sandbox:
                pyscript += '_sandbox'
            pyscript += '.py'

            py_command = 'python %s -a $add -d $d -e $e -x 1 -n $n -m $m -g $graph -s $strategy -dt $decay_type -nql $nq -eql $eq' % pyscript
            if backtrack:
                py_command += ' --backtrack'
            if one_way:
                py_command += ' --one_way'
            py_command += ' &\n'
            
            # logic for looping through parameter range and running the script
            f.write('for e in $(seq $emin $estep $emax); do\n')
            f.write('    for d in $(seq $dmin $dstep $dmax); do\n')
            f.write('        for iter in $(seq 1 1 $x); do\n')
            #f.write('            python ant_repair.py -a $add -d $d -e $e -x 1 -n $n -m $m -g $graph -s $strategy -dt $decay_type -nql $nq -eql $eq &\n')
            f.write('            ' + py_command)
            f.write('        done\n')
            # wait for all runs to finish before moving to next pair of parameter values
            f.write('        wait\n')
            f.write('    done\n')
            f.write('done\n')
            
            f.close()
