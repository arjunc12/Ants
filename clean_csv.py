from sys import argv
import os

def clean(in_file, out_file, n_cols, sep=','):
    in_file = open(in_file)
    out_file = open(out_file, 'w')
    for line in in_file:
        line = line.strip('\n')
        line = line.split(sep)
        if len(line) == n_cols:
            line = ','.join(line)
            out_file.write(line + '\n')
    in_file.close()
    out_file.close()
    
if __name__ == '__main__':
    in_file = argv[1]
    out_file = argv[2]
    n_cols = int(argv[3])
    clean(in_file, out_file, n_cols)