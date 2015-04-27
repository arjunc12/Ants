import os
import numpy as np
from sys import argv

def run_simulations(n, x, a_min, a_max, d_min, d_max, step, script):
    for add in np.arange(a_min, a_max + step, step):
        for decay in np.arange(d_min, d_max + step, step):
            command = "python %s -n %d -x %d -a %f -d %f" % (script, n, x, add, decay)
            os.system(command)
            
def main():
    n = int(argv[1])
    x = int(argv[2])
    a_min = float(argv[3])
    a_max = float(argv[4])
    d_min = float(argv[5])
    d_max = float(argv[6])
    step = float(argv[7])
    script = argv[8]
    if script == 'r':
        script = 'ant_rand_walk.py'
    else:
        script = 'ant_bfs.py'
    
    run_simulations(n, x, a_min, a_max, d_min, d_max, step, script)
    
if __name__ == '__main__':
    main()