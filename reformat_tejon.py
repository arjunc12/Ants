import pandas as pd
from sys import argv
import numpy as np

#choices = {1:('20', '24'), 2:('24', '20'), 3:('24', 'D'), 4:('D', '24'), 5:('24', 'B'),\
#           6:('B', '24'), 7:('24', 'C'), 8:('C', '24'), 9:('24', '17'), 10:('17', '24'),
#           11:('24', 'A'), 12:('A', '24')}

filename = argv[1]
infile = open(filename)
outfile = open('reformated_' + filename, 'w')
df = pd.read_csv(infile, header=None)
df[0] = pd.to_datetime(df[0])

edgeids = df[7]
edges = df[8]

edge_dic = {}

for i in xrange(len(edgeids)):
    edgeid = edgeids[i]
    edge = edges[i]
    if np.isnan(edgeid):
        break
    edgeid = int(edgeid)
    #print edge
    source, dest = edge.split(' to ')
    edge_dic[edgeid] = (source.strip(), dest.strip())
    
timestamps = df[0]
choices = df[1]
for i in xrange(len(timestamps)):
    timestamp = timestamps[i]
    choice = choices[i]
    try:
        choice = int(choice)
        source, dest = edge_dic[choice]
        source = source.replace(' ', '')
        dest = dest.replace(' ', '')
    except:
        continue
    outfile.write('%s, %s, %s\n' % (source, dest, str(timestamp)))
    
infile.close()
outfile.close()
    
'''
for line in infile:
    line = line.split(',')
    timestamp = line[0]
    timestamp = pd.to_datetime(timestamp)
    try:
        edge = int(line[1])
        source, dest = choices[edge]
    except:
        continue
    outfile.write('%s, %s, %s\n' % (source, dest, str(timestamp)))

infile.close()
outfile.close()
'''
