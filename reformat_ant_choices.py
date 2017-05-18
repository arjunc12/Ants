import pandas as pd
import datetime
import numpy as np
from sys import argv
import argparse

IN_DIR = 'datasets/csv'
OUT_DIR = 'datasets/reformated_csv'

def reformat_counts(label, delim=None):
    fname = 'counts%s.csv' % label
    in_file = open('%s/%s' % (IN_DIR, fname))
    out_file = open('%s/reformated_%s' % (OUT_DIR, fname), 'a')

    split_func = None
    if delim == None:
        split_func = lambda x : x.split()
    else:
        split_func = lambda x : x.split(delim)

    for line in in_file:
        line = line.strip('\n')
        line = line.split(',')
        if line[0] == 'Year':
            continue
        #print line
        year = int(line[0])
        day = int(line[1])
        hour = int(line[2])
        minute = int(line[3])
        second = float(line[4])
        dt = datetime.datetime(int(line[0]), 1, 1)
        delta = datetime.timedelta(days=day, hours=hour, minutes=minute, seconds=second)
        dt += delta
        dt = str(dt)
        edge = None
        for entry in line[5:]:
            #entry = entry.split('-')
            #entry = entry.split()
            entry = split_func(entry)
            #print entry
            if len(entry) == 2 and entry[0] != '':
                edge = entry
                break
        if edge == None:
            continue
        source = edge[0].strip()
        dest = edge[1].strip()
        out_file.write('%s, %s, %s\n' % (source, dest, dt))

    in_file.close()
    out_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', nargs='+')
    parser.add_argument('-d', '--delim', default=None)
    
    args = parser.parse_args()
    labels = args.labels
    delim = args.delim
    for label in labels:
        print label
        reformat_counts(label, delim)

if __name__ == '__main__':
    main()
