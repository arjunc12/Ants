import argparse

parser = argparse.ArgumentParser()

# required arguments
parser.add_argument('-s', '--strategies', nargs='+', required=True, dest='strategies')
parser.add_argument('-g', '--graphs', nargs='+', required=True, dest='graphs')
parser.add_argument('-dt', '--decay_types', nargs='+', required=True, dest='decay_types')
parser.add_argument('-m', '--steps', required=True, type=int, dest='steps')
parser.add_argument('-l', '--steps_label', required=True, dest='steps_label')

# optional arguments

# explore, decay ranges
parser.add_argument('-emin', type=float, default=0.05, dest='emin')
parser.add_argument('-emax', type=float, default=0.95, dest='emax')
parser.add_argument('-dmin', type=float, default=0.05, dest='dmin')
parser.add_argument('-dmax', type=float, default=0.95, dest='dmax')

# explore, decay increments
parser.add_argument('-estep', type=float, default=0.05, dest='estep')
parser.add_argument('-dstep', type=float, default=0.05, dest='dstep')

parser.add_argument('-a', '--add', type=float, default=1, dest='add')

parser.add_argument('-x', '--num_iters', type=int, default=50, dest='num_iters')
parser.add_argument('-n', '--num_ants', type=int, default=100, dest='num_ants')

parser.add_argument('-q', '--qlim', type=int, default=1, dest='qlim')

parser.add_argument('-e', '--explore', type=float, default=None, dest='explore')

args = parser.parse_args()
strategies = args.strategies
graphs = args.graphs
decay_types = args.decay_types
steps = args.steps
steps_label = args.steps_label

emin = args.emin
emax = args.emax
dmin = args.dmin
dmax = args.dmax

estep = args.estep
dstep = args.dstep

add = args.add

num_iters = args.num_iters
num_ants = args.num_ants

qlim = args.qlim

explore = args.explore

for strategy in strategies:
    for graph in graphs:
        for decay_type in decay_types:
            fname = 'ant_repair_sandbox_%s_%s_%s%s.sh' % (strategy, graph, decay_type, steps_label)
            print fname
            f = open(fname, 'w')
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

            f.write('q=%d\n' % qlim)
            f.write('\n')

            f.write('e=%f\n' % explore)
            f.write('\n')
            
            f.write('for e2 in $(seq $emin $estep $emax); do\n')
            f.write('    for d in $(seq $dmin $dstep $dmax); do\n')
            f.write('        for iter in $(seq 1 1 $x); do\n')
            f.write('            python ant_repair_sandbox.py -a $add -d $d -e $e -e2 $e2  -x 1 -n $n -m $m -g $graph -s $strategy -dt $decay_type -nql $q &\n')
            f.write('        done\n')
            f.write('        wait\n')
            f.write('    done\n')
            f.write('done\n')
            
            f.close()
