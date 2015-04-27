import os
import numpy as np
from sys import argv
import argparse

def run_simulations(script, n, x, a_min, a_max, d_min, d_max, step):
    for add in np.arange(a_min, a_max + step, step):
        for decay in np.arange(d_min, d_max + step, step):
            command = "python %s -n %d -x %d -a %f -d %f" % (script, n, x, add, decay)
            os.system(command)
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--repeats", type=int, dest="iterations", default=10,help="number of iterations")
    parser.add_argument("-a", "--add", type=float, dest="pheromone_add", nargs=2,help="amt of phermone added")
    parser.add_argument("-d", "--decay", type=float, dest="pheromone_decay", nargs=2,help="amt of pheromone decay")
    parser.add_argument("-n", "--number", type=int, dest="num_ants", default=10,help="number of ants")
    parser.add_argument("-s", "--step", type=float, dest="step_size", default=1, help="pheromone step size")
    parser.add_argument("-w", "--walk", dest="walk_type", choices=['r', 'b'], default='b', help="random walk or bfs")

    options = parser.parse_args()
    # ===============================================================

    # ===============================================================
    x = options.iterations
    a_min, a_max = options.pheromone_add
    d_min, d_max = options.pheromone_decay
    n = options.num_ants
    step = options.step_size
    walk = options.walk_type
    script = None
    if walk == 'b':
        script = "ant_bfs.py"
    else:
        script = "ant_rand_walk.py"
    
    run_simulations(script, n, x, a_min, a_max, d_min, d_max, step)
    
if __name__ == '__main__':
    main()