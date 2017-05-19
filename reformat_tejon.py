import pandas as pd
from sys import argv
import numpy as np
import argparse

#choices = {1:('20', '24'), 2:('24', '20'), 3:('24', 'D'), 4:('D', '24'), 5:('24', 'B'),\
#           6:('B', '24'), 7:('24', 'C'), 8:('C', '24'), 9:('24', '17'), 10:('17', '24'),
#           11:('24', 'A'), 12:('A', '24')}

IN_DIR = 'datasets/csv'
OUT_DIR = 'datasets/reformated_csv'

def reformat_tejon(label, edge_col=7, delim=None):
    split_func = None
    if delim == None:
        split_func = lambda x : x.split()
    else:
        split_func = lambda x : x.split(delim)
    
    infile = open('%s/counts%s.csv' % (IN_DIR, label))
    outfile = open('%s/reformated_counts%s.csv' % (OUT_DIR, label), 'w')
    df = pd.read_csv(infile, header=None)
    df[0] = pd.to_datetime(df[0])

    edgeids = df[edge_col]
    edges = df[edge_col + 1]

    edge_dic = {}

    for i in xrange(len(edgeids)):
        edgeid = edgeids[i]
        edge = edges[i]
        if np.isnan(edgeid):
            break
        edgeid = int(edgeid)
        source, dest = split_func(edge)
        edge_dic[edgeid] = (source.strip(), dest.strip())
    print edge_dic
       
    timestamps = df[0]
    choices = df[1]
    observed_choices = set()
    for i in xrange(len(timestamps)):
        timestamp = timestamps[i]
        choice = choices[i]
        try:
            choice = int(choice)
            observed_choices.add(choice)
            source, dest = edge_dic[choice]
            source = source.replace(' ', '')
            dest = dest.replace(' ', '')
        except:
            continue
        outfile.write('%s, %s, %s\n' % (source, dest, str(timestamp)))

    final_time = str(timestamps[len(timestamps) - 1])
    init_time = str(timestamps[0])
    for choice in edge_dic:
        if choice not in observed_choices:
            source, dest = edge_dic[choice]
            source = source.replace(' ', '')
            dest = dest.replace(' ', '')
            outfile.write('%s, %s, %s\n' % (source, dest, final_time))
        
    infile.close()
    outfile.close()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', nargs='+')
    parser.add_argument('-ec', '--edge_col', default=7, type=int)
    parser.add_argument('-d', '--delim', default=None)
    
    args = parser.parse_args()
    labels = args.labels
    edge_col = args.edge_col
    delim = args.delim

    for label in labels:
        print label
        reformat_tejon(label, edge_col, delim)

if __name__ == '__main__':
    main()
